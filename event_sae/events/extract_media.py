"""Render 5-frame bundle (PNGs + MP4 clip) around each AWE waypoint.

Inputs:
  - `waypoint_summary.json` from `event_sae.keyframes`.
  - The corresponding `trajectory_records.jsonl` (path is stored inside the
    waypoint summary).
  - `rollout.mp4`, or legacy episode videos under `<run_dir>/videos/`.

Outputs (under `output_dir`):
  - `frames/<sample_id>/frame_NN_stepMMMM.png` (one image per frame)
  - `clips/<sample_id>.mp4` (a short clip spanning the frame window)
  - `samples.jsonl` (per-sample manifest, one line per waypoint bundle)
  - `skipped_samples.jsonl` (waypoints that could not be rendered)
  - `packaging_report.json`

`samples.jsonl` is the input expected by `event_sae.events.build_features`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio


@dataclass
class EpisodeRecords:
    episode_num: int
    task_id: int
    task_episode_idx: int
    task_description: str
    prompt_task_description: str
    success: bool
    step_indices: list[int]


def _sanitize_label(label: str, max_len: int = 80) -> str:
    label = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")
    if not label:
        return "unnamed"
    return label[:max_len]


def load_waypoint_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "source_trajectory_records_path" not in data or "episodes" not in data:
        raise ValueError(f"Unexpected waypoint summary format: {path}")
    return data


def load_episode_records(path: Path) -> dict[int, EpisodeRecords]:
    grouped: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            grouped.setdefault(int(record["episode_num"]), []).append(record)

    episodes: dict[int, EpisodeRecords] = {}
    for episode_num, records in grouped.items():
        records.sort(key=lambda item: int(item["step_in_episode"]))
        first = records[0]
        episodes[episode_num] = EpisodeRecords(
            episode_num=episode_num,
            task_id=int(first["task_id"]),
            task_episode_idx=int(first["task_episode_idx"]),
            task_description=str(first["task_description"]),
            prompt_task_description=str(first.get("prompt_task_description", first["task_description"])),
            success=bool(records[-1]["done"]),
            step_indices=[int(record["step_in_episode"]) for record in records],
        )
    return episodes


def find_episode_video(run_dir: Path, episode_num: int) -> Path | None:
    """Locate ``rollout.mp4`` written alongside trajectory records by
    ``event_sae.rollout`` (the rollout entry point for both backends), or a legacy
    per-episode MP4 under ``videos/``:

    * OpenVLA: ``...--episode=<N>--...mp4``
    * OpenPI:  ``episode_<NNNNN>_task_...mp4`` (zero-padded).
    """
    if (run_dir / "rollout.mp4").is_file():
        result = json.loads((run_dir / "result.json").read_text())
        if int(result["settings"].get("episode_num", 0)) == episode_num:
            return run_dir / "rollout.mp4"
        return None
    videos_dir = run_dir / "videos"
    if not videos_dir.is_dir():
        return None
    for pattern in (
        f"*episode={episode_num}--*.mp4",       # OpenVLA
        f"episode_{episode_num:05d}_*.mp4",     # OpenPI
    ):
        matches = sorted(videos_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def fit_frame_window(
    *,
    waypoint_index: int,
    frame_offsets: list[int],
    num_frames: int,
) -> tuple[list[int] | None, list[int], int | None]:
    """Shift a fixed-stride frame window so it stays within [0, num_frames).

    Returns (frame_indices, requested_frame_indices, window_shift).
    `frame_indices=None` if the window span exceeds the episode length.
    """
    requested_frame_indices = [waypoint_index + offset for offset in frame_offsets]
    if not requested_frame_indices:
        return [], [], 0

    min_requested = min(requested_frame_indices)
    max_requested = max(requested_frame_indices)
    if (max_requested - min_requested) >= num_frames:
        return None, requested_frame_indices, None

    shift = 0
    if max_requested >= num_frames:
        shift -= max_requested - (num_frames - 1)
    if min_requested + shift < 0:
        shift += -(min_requested + shift)

    frame_indices = [index + shift for index in requested_frame_indices]
    return frame_indices, requested_frame_indices, shift


def _save_frame(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame)


def _write_clip(reader, frame_start: int, frame_end: int, fps: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for frame_idx in range(frame_start, frame_end + 1):
            writer.append_data(reader.get_data(frame_idx))
    finally:
        writer.close()


def _write_skipped_episode_waypoints(
    skipped_file,
    *,
    reason: str,
    episode: dict,
    episode_num: int,
    video_path: Path | None = None,
    error: Exception | None = None,
) -> int:
    skipped_count = 0
    for waypoint_rank, waypoint_index in enumerate(episode.get("waypoint_indices", [])):
        record = {
            "reason": reason,
            "episode_num": episode_num,
            "waypoint_rank": waypoint_rank,
            "waypoint_index": int(waypoint_index),
        }
        if video_path is not None:
            record["video_path"] = str(video_path)
        if error is not None:
            record["error"] = repr(error)
        skipped_file.write(json.dumps(record) + "\n")
        skipped_count += 1
    return skipped_count


def extract_keyframe_media(
    waypoint_summary_path: Path,
    output_dir: Path,
    frame_offsets: list[int],
    max_samples: int | None = None,
) -> dict:
    """Render frame PNGs + clip MP4s around each waypoint and write `samples.jsonl`.

    Returns the packaging report dict (also written to `packaging_report.json`).
    """
    waypoint_summary_path = Path(waypoint_summary_path).resolve()
    output_dir = Path(output_dir).resolve()
    summary = load_waypoint_summary(waypoint_summary_path)

    trajectory_records_path = Path(summary["source_trajectory_records_path"]).resolve()
    if not trajectory_records_path.is_file():
        raise FileNotFoundError(f"trajectory_records.jsonl not found: {trajectory_records_path}")
    run_dir = trajectory_records_path.parent
    episode_records = load_episode_records(trajectory_records_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_root = output_dir / "frames"
    clips_root = output_dir / "clips"
    samples_path = output_dir / "samples.jsonl"
    skipped_path = output_dir / "skipped_samples.jsonl"

    sample_count = 0
    skipped_count = 0
    with samples_path.open("w", encoding="utf-8") as samples_file, skipped_path.open(
        "w", encoding="utf-8"
    ) as skipped_file:
        for episode in summary["episodes"]:
            if max_samples is not None and sample_count >= max_samples:
                break

            episode_num = int(episode["episode_num"])
            episode_meta = episode_records.get(episode_num)
            if episode_meta is None:
                skipped_file.write(
                    json.dumps({"reason": "missing_episode_records", "episode_num": episode_num}) + "\n"
                )
                skipped_count += 1
                continue

            video_path = find_episode_video(run_dir, episode_num)
            if video_path is None:
                skipped_file.write(
                    json.dumps({"reason": "missing_episode_video", "episode_num": episode_num}) + "\n"
                )
                skipped_count += 1
                continue

            reader = None
            try:
                reader = imageio.get_reader(video_path)
                fps = float(reader.get_meta_data().get("fps", 20.0))
            except Exception as exc:
                skipped_count += _write_skipped_episode_waypoints(
                    skipped_file,
                    reason="unreadable_episode_video",
                    episode=episode,
                    episode_num=episode_num,
                    video_path=video_path,
                    error=exc,
                )
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:
                        pass
                continue

            try:
                num_frames = len(episode_meta.step_indices)
                for waypoint_rank, waypoint_index in enumerate(episode["waypoint_indices"]):
                    if max_samples is not None and sample_count >= max_samples:
                        break

                    waypoint_index = int(waypoint_index)
                    frame_indices, requested_frame_indices, window_shift = fit_frame_window(
                        waypoint_index=waypoint_index,
                        frame_offsets=frame_offsets,
                        num_frames=num_frames,
                    )
                    if frame_indices is None:
                        skipped_file.write(
                            json.dumps(
                                {
                                    "reason": "window_span_exceeds_episode",
                                    "episode_num": episode_num,
                                    "waypoint_rank": waypoint_rank,
                                    "waypoint_index": waypoint_index,
                                    "requested_frame_indices": requested_frame_indices,
                                    "num_frames": num_frames,
                                }
                            )
                            + "\n"
                        )
                        skipped_count += 1
                        continue

                    sample_id = (
                        f"ep{episode_num:04d}_task{episode_meta.task_id:02d}_"
                        f"wp{waypoint_rank:02d}_step{episode_meta.step_indices[waypoint_index]:04d}_"
                        f"{_sanitize_label(episode_meta.task_description, max_len=36)}"
                    )
                    sample_frame_dir = frames_root / sample_id
                    frame_paths: list[str] = []
                    frame_steps = [episode_meta.step_indices[idx] for idx in frame_indices]
                    for frame_pos, (frame_index, frame_step) in enumerate(
                        zip(frame_indices, frame_steps, strict=True)
                    ):
                        frame = reader.get_data(frame_index)
                        frame_path = sample_frame_dir / f"frame_{frame_pos:02d}_step{frame_step:04d}.png"
                        _save_frame(frame, frame_path)
                        frame_paths.append(str(frame_path))

                    clip_path = clips_root / f"{sample_id}.mp4"
                    _write_clip(
                        reader=reader,
                        frame_start=min(frame_indices),
                        frame_end=max(frame_indices),
                        fps=fps,
                        output_path=clip_path,
                    )

                    sample_record = {
                        "sample_id": sample_id,
                        "run_dir": str(run_dir),
                        "source_waypoint_summary_path": str(waypoint_summary_path),
                        "source_trajectory_records_path": str(trajectory_records_path),
                        "source_video_path": str(video_path),
                        "episode_num": episode_num,
                        "task_id": episode_meta.task_id,
                        "task_suite_name": episode.get("task_suite_name"),
                        "task_episode_idx": episode_meta.task_episode_idx,
                        "task_description": episode_meta.task_description,
                        "prompt_task_description": episode_meta.prompt_task_description,
                        "success": episode_meta.success,
                        "waypoint_rank": waypoint_rank,
                        "waypoint_index": waypoint_index,
                        "waypoint_step": episode_meta.step_indices[waypoint_index],
                        "frame_offsets": list(frame_offsets),
                        "requested_frame_indices": requested_frame_indices,
                        "frame_indices": frame_indices,
                        "frame_steps": frame_steps,
                        "window_shift": window_shift,
                        "frame_paths": frame_paths,
                        "clip_path": str(clip_path),
                    }
                    samples_file.write(json.dumps(sample_record) + "\n")
                    samples_file.flush()
                    sample_count += 1
            finally:
                reader.close()

    report = {
        "waypoint_summary_path": str(waypoint_summary_path),
        "source_trajectory_records_path": str(trajectory_records_path),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "frame_offsets": list(frame_offsets),
        "num_samples": sample_count,
        "num_skipped": skipped_count,
        "samples_path": str(samples_path),
        "skipped_samples_path": str(skipped_path),
    }
    report_path = output_dir / "packaging_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
