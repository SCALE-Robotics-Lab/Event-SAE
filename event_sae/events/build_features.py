"""Build per-event vision embedding + state vector for each sample.

Reads `samples.jsonl` from `event_sae.events.extract_media`, encodes the
selected frames through a frozen vision encoder (default SigLIP base), and
joins per-waypoint end-effector position + (optional) gripper action.

Output: `event_features.jsonl` (one record per sample) ready for
`event_sae.events.cluster`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from event_sae.events.io import load_jsonl, write_jsonl


@dataclass
class EpisodeStateSummary:
    num_steps: int
    center_records: dict[int, dict]
    state_feature_names: list[str]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("Encountered zero-norm vector during event feature construction.")
    return vec / norm


class VisionEmbedder:
    def __init__(self, model_name_or_path: str, device: str, revision: str | None = None) -> None:
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, revision=revision)
        self.model = AutoModel.from_pretrained(model_name_or_path, revision=revision).eval().to(device)
        self.device = torch.device(device)

    @torch.no_grad()
    def encode(self, frame_paths: list[str]) -> np.ndarray:
        images = [Image.open(path).convert("RGB") for path in frame_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        if hasattr(self.model, "get_image_features"):
            feats = self.model.get_image_features(**inputs)
        else:
            outputs = self.model(**inputs)
            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                feats = outputs.image_embeds
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                feats = outputs.last_hidden_state.mean(dim=1)
            else:
                raise ValueError(
                    "Could not derive image features from the selected vision model outputs."
                )
        mean_feat = feats.float().mean(dim=0).cpu().numpy()
        return _l2_normalize(mean_feat)


def build_episode_state_index(samples: list[dict]) -> dict[tuple[str, int], EpisodeStateSummary]:
    """Scan each trajectory_records.jsonl referenced by samples and pre-collect
    per-waypoint center states (eef_pos + optional gripper_action)."""
    needed_steps: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for sample in samples:
        trajectory_path = str(Path(sample["source_trajectory_records_path"]).resolve())
        needed_steps[trajectory_path][int(sample["episode_num"])].add(int(sample["waypoint_step"]))

    index: dict[tuple[str, int], EpisodeStateSummary] = {}
    for trajectory_path, episode_steps in needed_steps.items():
        center_records: dict[int, dict[int, dict]] = {
            episode_num: {} for episode_num in episode_steps
        }
        step_counts: dict[int, int] = defaultdict(int)
        gripper_action_presence: set[bool] = set()
        with Path(trajectory_path).open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                episode_num = int(record["episode_num"])
                if episode_num not in episode_steps:
                    continue
                step = int(record["step_in_episode"])
                step_counts[episode_num] += 1
                if step in episode_steps[episode_num]:
                    if "eef_pos" not in record:
                        raise KeyError(
                            f"Missing eef_pos for episode {episode_num}, step {step} in {trajectory_path}"
                        )
                    has_gripper_action = "gripper_action" in record
                    gripper_action_presence.add(has_gripper_action)
                    center_record = {"eef_pos": [float(x) for x in record["eef_pos"]]}
                    if has_gripper_action:
                        center_record["gripper_action"] = float(record["gripper_action"])
                    center_records[episode_num][step] = center_record

        if len(gripper_action_presence) > 1:
            raise ValueError(
                "Inconsistent gripper_action availability across selected keyframe steps "
                f"in {trajectory_path}"
            )
        state_feature_names = ["eef_pos_x", "eef_pos_y", "eef_pos_z"]
        if gripper_action_presence == {True}:
            state_feature_names.append("gripper_action")

        for episode_num, step_map in episode_steps.items():
            missing_steps = sorted(step_map.difference(center_records[episode_num]))
            if missing_steps:
                raise ValueError(
                    f"Missing center states for episode {episode_num} steps {missing_steps[:10]} in {trajectory_path}"
                )
            index[(trajectory_path, episode_num)] = EpisodeStateSummary(
                num_steps=int(step_counts[episode_num]),
                center_records=center_records[episode_num],
                state_feature_names=state_feature_names,
            )
    return index


def state_vector_from_record(center_state: dict, state_feature_names: list[str]) -> np.ndarray:
    values = [
        center_state["eef_pos"][0],
        center_state["eef_pos"][1],
        center_state["eef_pos"][2],
    ]
    if "gripper_action" in state_feature_names:
        values.append(center_state["gripper_action"])
    return np.asarray(values, dtype=np.float32)


def build_event_features(
    samples_path: Path,
    output_path: Path,
    vision_model_name_or_path: str = "google/siglip-base-patch16-224",
    device: str | None = None,
    frame_positions: list[int] = (0, 1, 2, 3, 4),
    vision_revision: str | None = None,
) -> None:
    """Compute vision embeddings + state vectors per sample and write event_features.jsonl."""
    samples_path = Path(samples_path).resolve()
    if not samples_path.is_file():
        raise FileNotFoundError(f"samples.jsonl not found: {samples_path}")
    output_path = Path(output_path).resolve()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_jsonl(samples_path)
    state_index = build_episode_state_index(samples)
    embedder = VisionEmbedder(vision_model_name_or_path, device, revision=vision_revision)

    records: list[dict] = []
    for idx, sample in enumerate(samples, start=1):
        frame_paths = sample["frame_paths"]
        selected_frame_paths = [frame_paths[pos] for pos in frame_positions]
        trajectory_path = str(Path(sample["source_trajectory_records_path"]).resolve())
        episode_num = int(sample["episode_num"])
        waypoint_step = int(sample["waypoint_step"])
        state_summary = state_index[(trajectory_path, episode_num)]
        center_state = state_summary.center_records[waypoint_step]
        progress_percent = waypoint_step / max(state_summary.num_steps - 1, 1)
        vision_embedding = embedder.encode(selected_frame_paths)
        state_vector = state_vector_from_record(
            center_state=center_state,
            state_feature_names=state_summary.state_feature_names,
        )
        records.append(
            {
                "sample_id": sample["sample_id"],
                "task_id": int(sample["task_id"]),
                "task_suite_name": sample.get("task_suite_name"),
                "task_description": sample["task_description"],
                "prompt_task_description": sample["prompt_task_description"],
                "episode_num": episode_num,
                "task_episode_idx": int(sample["task_episode_idx"]),
                "waypoint_rank": int(sample["waypoint_rank"]),
                "waypoint_step": waypoint_step,
                "clip_path": sample["clip_path"],
                "frame_paths": frame_paths,
                "selected_frame_paths": selected_frame_paths,
                "source_trajectory_records_path": trajectory_path,
                "vision_model_name_or_path": vision_model_name_or_path,
                "vision_revision": vision_revision,
                "vision_frame_positions": list(frame_positions),
                "vision_embedding": vision_embedding.astype(np.float32).tolist(),
                "state_vector": state_vector.tolist(),
                "state_feature_names": state_summary.state_feature_names,
                "progress_percent": float(progress_percent),
                "num_steps": int(state_summary.num_steps),
            }
        )
        print(
            f"[{idx}/{len(samples)}] sample_id={sample['sample_id']} "
            f"task={sample['task_description']} progress={progress_percent:.3f}"
        )

    write_jsonl(output_path, records)
    print(f"Samples path: {samples_path}")
    print(f"Saved event features to: {output_path}")
