"""Dense shard writing and metadata helpers for OpenPI SAE collection."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

FORWARD_TYPE_NAMES = {
    0: "prefix_prefill",
    1: "denoise_step",
}


@dataclasses.dataclass(frozen=True)
class CaptureTargetSpec:
    name: str
    expert_idx: int
    layer_prefix: str
    description: str


CAPTURE_TARGETS: dict[str, CaptureTargetSpec] = {
    "paligemma": CaptureTargetSpec(
        name="paligemma",
        expert_idx=0,
        layer_prefix="PG",
        description="PaliGemma/prefix stream after the block MLP residual add.",
    ),
    "action_expert": CaptureTargetSpec(
        name="action_expert",
        expert_idx=1,
        layer_prefix="AE",
        description="Action-expert/suffix stream after the block MLP residual add.",
    ),
}


def parse_layer_indices(raw: str | int | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(raw, int):
        return (raw,)
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in raw)
    return tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())


def make_layer_mask(layer_indices: tuple[int, ...], depth: int) -> np.ndarray:
    mask = np.zeros((depth,), dtype=np.bool_)
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= depth:
            raise ValueError(f"Layer index {layer_idx} is outside model depth {depth}.")
        mask[layer_idx] = True
    return mask


def sanitize_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "run"


def ensure_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ensure_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        if math.isnan(as_float) or math.isinf(as_float):
            return None
        return as_float
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return ensure_jsonable(value.detach().cpu().numpy())
    return value


@dataclasses.dataclass(frozen=True)
class SAECollectionConfig:
    output_root: str
    run_name: str | None
    capture_target: str
    layer_indices: tuple[int, ...]
    flush_every_rows: int = 50_000
    submodule_name: str = "resid_post"
    schema_version: str = "openpi_sae_collection_v1"
    append_to_existing: bool = False
    model_type: str | None = None
    pi05: bool | None = None
    action_horizon: int | None = None
    action_dim: int | None = None
    state_token_offset: int | None = None
    action_token_offset: int | None = None
    replan_steps: int | None = None

    def target_spec(self) -> CaptureTargetSpec:
        if self.capture_target not in CAPTURE_TARGETS:
            raise ValueError(
                f"Unsupported capture_target={self.capture_target!r}. " f"Expected one of {sorted(CAPTURE_TARGETS)}."
            )
        return CAPTURE_TARGETS[self.capture_target]


@dataclasses.dataclass
class _PendingRecord:
    context: dict[str, Any]
    forward_type: str
    forward_step_idx: int
    global_forward_idx: int
    layer_idx: int
    layer_name: str
    batch_size: int
    seq_len: int
    d_model: int
    row_start: int
    row_end: int
    predicted_action_first: list[float] | None = None
    gripper_action: float | None = None
    action_chunk_shape: list[int] | None = None
    infer_ms: float | None = None


@dataclasses.dataclass
class _LayerBuffer:
    chunks: list[torch.Tensor] = dataclasses.field(default_factory=list)
    pending_records: list[_PendingRecord] = dataclasses.field(default_factory=list)
    row_count: int = 0
    shard_id: int = 0
    total_rows: int = 0


@dataclasses.dataclass
class _StagedRecord:
    context: dict[str, Any]
    forward_type: str
    forward_step_idx: int
    global_forward_idx: int
    layer_idx: int
    layer_name: str
    batch_size: int
    seq_len: int
    d_model: int
    num_rows: int
    predicted_action_first: list[float] | None = None
    gripper_action: float | None = None
    action_chunk_shape: list[int] | None = None
    infer_ms: float | None = None


@dataclasses.dataclass
class _EpisodeLayerBuffer:
    chunks: list[torch.Tensor] = dataclasses.field(default_factory=list)
    records: list[_StagedRecord] = dataclasses.field(default_factory=list)
    row_count: int = 0


@dataclasses.dataclass
class _EpisodeState:
    episode_key: str
    episode_num: int
    layer_buffers: dict[int, _EpisodeLayerBuffer]
    request_count: int = 0


@dataclasses.dataclass
class _RequestState:
    context: dict[str, Any]
    created_at: str
    episode_key: str
    forward_ids: dict[tuple[str, int], int] = dataclasses.field(default_factory=dict)
    records: list[_StagedRecord] = dataclasses.field(default_factory=list)


class ActivationCollector:
    """Collect dense post-MLP residual activations into SAE-compatible shards."""

    def __init__(
        self,
        config: SAECollectionConfig,
        *,
        policy_config_name: str,
        checkpoint_dir: str,
        model_depth: int,
        d_model: int,
    ) -> None:
        self._config = config
        self._policy_config_name = policy_config_name
        self._checkpoint_dir = checkpoint_dir
        self._model_depth = int(model_depth)
        self._d_model = int(d_model)
        self._target = config.target_spec()
        self._layer_mask = make_layer_mask(config.layer_indices, self._model_depth)
        self._selected_layers = tuple(idx for idx in range(self._model_depth) if self._layer_mask[idx])
        self._global_forward_idx = 0
        self._sample_hidden_shapes: dict[str, list[int]] = {}
        self._observed_executed_chunk_lens: set[int] = set()
        self._request_state: _RequestState | None = None
        self._staged_episodes: dict[str, _EpisodeState] = {}

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = sanitize_name(config.run_name or f"{policy_config_name}_{self._target.name}_{timestamp}")
        output_root = Path(config.output_root)
        base_run_dir = output_root / run_name
        if config.append_to_existing:
            self.run_dir = base_run_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.run_dir = self._dedupe_run_dir(base_run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=False)

        if self._target.name == "action_expert":
            acts_dir = self.run_dir / "sae_activations" / config.submodule_name
        else:
            acts_dir = self.run_dir / "sae_activations" / f"{config.submodule_name}__{self._target.name}"
        acts_dir.mkdir(parents=True, exist_ok=True)
        self._activations_dir = acts_dir

        self._activation_index_path = self.run_dir / "activation_index.jsonl"
        existing_index_records = _load_jsonl(self._activation_index_path) if config.append_to_existing else []
        self._activation_index_file = self._activation_index_path.open(
            "a" if config.append_to_existing else "w",
            encoding="utf-8",
        )
        self._log_path = self.run_dir / "collection.log"
        self._log_file = self._log_path.open("a" if config.append_to_existing else "w", encoding="utf-8")
        self._layers = {layer_idx: _LayerBuffer() for layer_idx in self._selected_layers}
        self._restore_existing_state(existing_index_records)

        self._write_log(
            "Initialized SAE collector: "
            f"run_dir={self.run_dir} target={self._target.name} layers={list(self._selected_layers)} "
            f"flush_every_rows={self._config.flush_every_rows} append_to_existing={self._config.append_to_existing}"
        )
        self._write_run_metadata()
        self._write_notes()

    @property
    def activation_index_path(self) -> Path:
        return self._activation_index_path

    @property
    def capture_target(self) -> str:
        return self._target.name

    @property
    def expert_idx(self) -> int:
        return self._target.expert_idx

    @property
    def selected_layers(self) -> tuple[int, ...]:
        return self._selected_layers

    @property
    def layer_mask(self) -> np.ndarray:
        return self._layer_mask.copy()

    def server_metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "schema_version": self._config.schema_version,
            "run_dir": str(self.run_dir.resolve()),
            "capture_target": self._target.name,
            "layer_indices": list(self._selected_layers),
            "activation_index_path": str(self._activation_index_path.resolve()),
            "submodule_name": self._config.submodule_name,
            "model_type": self._config.model_type,
            "pi05": self._config.pi05,
            "action_horizon": self._config.action_horizon,
            "action_dim": self._config.action_dim,
            "state_token_offset": self._config.state_token_offset,
            "action_token_offset": self._config.action_token_offset,
            "replan_steps": self._config.replan_steps,
        }

    def begin_request(self, context: dict[str, Any]) -> None:
        if self._request_state is not None:
            raise RuntimeError("A collection request is already active.")
        request_context = ensure_jsonable(context)
        if request_context.get("executed_chunk_len") is not None:
            self._observed_executed_chunk_lens.add(int(request_context["executed_chunk_len"]))
        request_context.setdefault("capture_target", self._target.name)
        request_context.setdefault("capture_submodule_name", self._config.submodule_name)
        episode_key = self._episode_key_from_context(request_context)
        if episode_key not in self._staged_episodes:
            self._staged_episodes[episode_key] = _EpisodeState(
                episode_key=episode_key,
                episode_num=int(request_context["episode_num"]),
                layer_buffers={layer_idx: _EpisodeLayerBuffer() for layer_idx in self._selected_layers},
            )
        self._staged_episodes[episode_key].request_count += 1
        self._request_state = _RequestState(
            context=request_context,
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
            episode_key=episode_key,
        )

    def clear_request(self) -> None:
        self._request_state = None

    def finish_request(self, outputs: dict[str, Any]) -> None:
        if self._request_state is None:
            return

        actions = outputs.get("actions")
        action_array = None if actions is None else np.asarray(actions)
        action_first = None
        gripper_action = None
        action_chunk_shape = None
        if action_array is not None:
            action_chunk_shape = list(action_array.shape)
            if action_array.ndim >= 2 and action_array.shape[0] > 0:
                action_first = ensure_jsonable(action_array[0])
                if action_array.shape[-1] > 0:
                    gripper_action = float(action_array[0, -1])
            elif action_array.ndim == 1:
                action_first = ensure_jsonable(action_array)
                gripper_action = float(action_array[-1])

        infer_ms = None
        policy_timing = outputs.get("policy_timing")
        if isinstance(policy_timing, dict) and policy_timing.get("infer_ms") is not None:
            infer_ms = float(policy_timing["infer_ms"])

        for record in self._request_state.records:
            record.predicted_action_first = action_first
            record.gripper_action = gripper_action
            record.action_chunk_shape = action_chunk_shape
            record.infer_ms = infer_ms

    def record_activation(
        self,
        *,
        layer_idx: int,
        expert_idx: int,
        forward_type: int,
        forward_step_idx: int,
        hidden_states: np.ndarray,
    ) -> None:
        if self._request_state is None:
            return
        if expert_idx != self._target.expert_idx:
            return
        if layer_idx not in self._layers:
            return
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected hidden_states ndim=3 for dense SAE capture, got shape={tuple(hidden_states.shape)}"
            )

        batch_size, seq_len, d_model = hidden_states.shape
        if d_model != self._d_model:
            self._write_log(
                f"Observed d_model={d_model} for layer {layer_idx}, collector initialized with d_model={self._d_model}"
            )
        layer_key = f"layer_{layer_idx:02d}"
        if layer_key not in self._sample_hidden_shapes:
            self._sample_hidden_shapes[layer_key] = [int(batch_size), int(seq_len), int(d_model)]
            self._write_log(f"Sample hidden shape {layer_key}: batch={batch_size} seq_len={seq_len} d_model={d_model}")

        forward_name = FORWARD_TYPE_NAMES.get(forward_type, f"forward_type_{forward_type}")
        forward_key = (forward_name, int(forward_step_idx))
        if forward_key not in self._request_state.forward_ids:
            self._global_forward_idx += 1
            self._request_state.forward_ids[forward_key] = self._global_forward_idx
        global_forward_idx = self._request_state.forward_ids[forward_key]

        sample_np = np.asarray(hidden_states, dtype=np.float32).reshape(-1, hidden_states.shape[-1]).copy()
        sample = torch.from_numpy(sample_np).cpu()
        episode_state = self._staged_episodes[self._request_state.episode_key]
        episode_layer_buffer = episode_state.layer_buffers[layer_idx]
        episode_layer_buffer.chunks.append(sample)
        episode_layer_buffer.row_count += int(sample.shape[0])

        request_context = dict(self._request_state.context)
        request_context.setdefault("global_forward_idx", global_forward_idx)
        request_context.setdefault("forward_type", forward_name)
        request_context.setdefault("forward_step_idx", int(forward_step_idx))
        request_context.setdefault("forward_created_at", self._request_state.created_at)

        record = _StagedRecord(
            context=request_context,
            forward_type=forward_name,
            forward_step_idx=int(forward_step_idx),
            global_forward_idx=global_forward_idx,
            layer_idx=int(layer_idx),
            layer_name=f"{self._target.layer_prefix}{layer_idx:02d}",
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            d_model=int(d_model),
            num_rows=int(sample.shape[0]),
        )
        self._request_state.records.append(record)
        episode_layer_buffer.records.append(record)

    def handle_control(self, command: dict[str, Any]) -> dict[str, Any]:
        command_name = command.get("command")
        if command_name != "finalize_episode":
            return {
                "status": "error",
                "message": f"Unsupported SAE collection control command: {command_name!r}",
            }

        episode_num = command.get("episode_num")
        keep = command.get("keep")
        if episode_num is None or keep is None:
            return {
                "status": "error",
                "message": "finalize_episode requires both episode_num and keep.",
            }
        return self.finalize_episode(int(episode_num), bool(keep))

    def finalize_episode(self, episode_num: int, keep: bool) -> dict[str, Any]:
        episode_key = str(int(episode_num))
        episode_state = self._staged_episodes.pop(episode_key, None)
        if episode_state is None:
            return {
                "status": "error",
                "message": f"No staged SAE data found for episode_num={episode_num}.",
            }

        activation_records = 0
        num_rows = 0
        if keep:
            for layer_idx in self._selected_layers:
                episode_layer_buffer = episode_state.layer_buffers[layer_idx]
                layer_buffer = self._layers[layer_idx]
                for chunk, record in zip(episode_layer_buffer.chunks, episode_layer_buffer.records, strict=True):
                    row_start = layer_buffer.row_count
                    row_end = row_start + int(chunk.shape[0])
                    layer_buffer.chunks.append(chunk)
                    layer_buffer.row_count = row_end
                    layer_buffer.pending_records.append(
                        _PendingRecord(
                            context=record.context,
                            forward_type=record.forward_type,
                            forward_step_idx=record.forward_step_idx,
                            global_forward_idx=record.global_forward_idx,
                            layer_idx=record.layer_idx,
                            layer_name=record.layer_name,
                            batch_size=record.batch_size,
                            seq_len=record.seq_len,
                            d_model=record.d_model,
                            row_start=int(row_start),
                            row_end=int(row_end),
                            predicted_action_first=record.predicted_action_first,
                            gripper_action=record.gripper_action,
                            action_chunk_shape=record.action_chunk_shape,
                            infer_ms=record.infer_ms,
                        )
                    )
                    activation_records += 1
                    num_rows += int(chunk.shape[0])
                if layer_buffer.row_count >= self._config.flush_every_rows:
                    self._flush_layer(layer_idx)
            self._write_log(f"Kept episode {episode_num}: activation_records={activation_records} num_rows={num_rows}")
        else:
            for layer_idx in self._selected_layers:
                episode_layer_buffer = episode_state.layer_buffers[layer_idx]
                activation_records += len(episode_layer_buffer.records)
                num_rows += sum(int(chunk.shape[0]) for chunk in episode_layer_buffer.chunks)
            self._write_log(
                f"Discarded episode {episode_num}: activation_records={activation_records} num_rows={num_rows}"
            )

        return {
            "status": "ok",
            "episode_num": int(episode_num),
            "kept": bool(keep),
            "activation_records": int(activation_records),
            "num_rows": int(num_rows),
        }

    def close(self) -> None:
        if self._staged_episodes:
            dropped = [state.episode_num for state in self._staged_episodes.values()]
            self._write_log(f"Dropping undecided staged episodes on close: {sorted(dropped)}")
            self._staged_episodes.clear()
        for layer_idx in self._selected_layers:
            self._flush_layer(layer_idx)
        self._activation_index_file.close()
        self._write_run_metadata()
        self._write_notes()
        self._write_log("Collector closed.")
        self._log_file.close()

    def _flush_layer(self, layer_idx: int) -> None:
        layer_buffer = self._layers[layer_idx]
        if not layer_buffer.chunks:
            return

        activations = torch.cat(layer_buffer.chunks, dim=0)
        shard_path = self._activations_dir / f"layer_{layer_idx:02d}_shard_{layer_buffer.shard_id:06d}.pt"
        torch.save(activations, shard_path)

        relative_shard_path = str(shard_path.relative_to(self.run_dir))
        for record in layer_buffer.pending_records:
            payload = {
                **record.context,
                "capture_target": self._target.name,
                "capture_submodule_name": self._config.submodule_name,
                "layer_idx": record.layer_idx,
                "layer_name": record.layer_name,
                "shard_path": relative_shard_path,
                "row_start": record.row_start,
                "row_end": record.row_end,
                "seq_len": record.seq_len,
                "batch_size": record.batch_size,
                "d_model": record.d_model,
                "forward_type": record.forward_type,
                "forward_step_idx": record.forward_step_idx,
                "global_forward_idx": record.global_forward_idx,
                "predicted_action_first": record.predicted_action_first,
                "gripper_action": record.gripper_action,
                "action_chunk_shape": record.action_chunk_shape,
                "infer_ms": record.infer_ms,
            }
            self._activation_index_file.write(json.dumps(ensure_jsonable(payload)) + "\n")
        self._activation_index_file.flush()

        layer_buffer.total_rows += int(activations.shape[0])
        layer_buffer.shard_id += 1
        layer_buffer.chunks.clear()
        layer_buffer.pending_records.clear()
        layer_buffer.row_count = 0
        self._write_log(
            f"Wrote {relative_shard_path} shape={tuple(activations.shape)} total_rows={layer_buffer.total_rows}"
        )

    def _write_log(self, message: str) -> None:
        timestamp = dt.datetime.now().isoformat(timespec="seconds")
        self._log_file.write(f"[{timestamp}] {message}\n")
        self._log_file.flush()

    def _episode_key_from_context(self, context: dict[str, Any]) -> str:
        if context.get("episode_num") is None:
            raise ValueError(
                "SAE collection requires episode_num in the request context so episodes can be kept or discarded."
            )
        return str(int(context["episode_num"]))

    def _write_notes(self) -> None:
        notes_path = self.run_dir / "notes.md"
        hook_file = "third_party/openpi/src/openpi/models/gemma.py"
        hook_expr = "xs = [_gated_residual(x, y, gate) ...] immediately after the MLP residual add in Block.__call__"
        content = f"""# OpenPI SAE Collection Notes

- Hook site: `{hook_file}` at the post-MLP residual add inside `Block.__call__`.
- Primary analogue: `{self._config.submodule_name}` from the `{self._target.name}` stream (`expert_idx={self._target.expert_idx}`).
- Why this site: it is the hidden state after attention, after MLP, and after the residual add, which matches the OpenVLA decoder-layer output captured by the existing `resid_post` pipeline.
- Flattening rule: each captured tensor is reshaped from `(batch, seq_len, d_model)` to `(batch * seq_len, d_model)` before buffering.
- Shard layout: dense tensors are written under `{self._activations_dir.relative_to(self.run_dir)}` as `layer_XX_shard_XXXXXX.pt`.
- Index rows: `row_start` is inclusive, `row_end` is exclusive, and each span refers to rows inside the saved shard file for that layer.
- Metadata path: `activation_index.jsonl` maps `(prompt/task, episode, step/timestamp)` back to `(layer, shard_path, row_start, row_end, seq_len, global_forward_idx)`.
- Training compatibility: the primary `{self._target.name}` collection keeps the existing flat `sae_activations/{self._config.submodule_name}/layer_XX_shard_*.pt` layout expected by the current SAE training code.
- Optional mismatch note: unlike OpenVLA's single decoder stack, pi0.5 has separate PaliGemma and action-expert streams. This run captures `{self._target.name}` because it is the stream directly used for action generation in the fine-tuned checkpoint.

## Lookup

Given `prompt/task`, `episode_num`, and `step_in_episode` (or timestamp), filter `activation_index.jsonl` to the matching entries. Each matching entry includes:

- `layer_idx`
- `layer_name`
- `shard_path`
- `row_start`
- `row_end`
- `seq_len`
- `global_forward_idx`
"""
        notes_path.write_text(content, encoding="utf-8")

    def _write_run_metadata(self) -> None:
        metadata_path = self.run_dir / "run_metadata.json"
        payload = {
            "schema_version": self._config.schema_version,
            "policy_config_name": self._policy_config_name,
            "checkpoint_dir": self._checkpoint_dir,
            "run_dir": str(self.run_dir),
            "capture_target": self._target.name,
            "capture_target_description": self._target.description,
            "capture_submodule_name": self._config.submodule_name,
            "layer_indices": list(self._selected_layers),
            "model_depth": self._model_depth,
            "d_model": self._d_model,
            "model_type": self._config.model_type,
            "pi05": self._config.pi05,
            "action_horizon": self._config.action_horizon,
            "action_dim": self._config.action_dim,
            "state_token_offset": self._config.state_token_offset,
            "action_token_offset": self._config.action_token_offset,
            "replan_steps": self._config.replan_steps,
            "observed_executed_chunk_lens": sorted(self._observed_executed_chunk_lens),
            "token_layout": {
                "action_horizon": self._config.action_horizon,
                "action_dim": self._config.action_dim,
                "state_token_count": self._config.state_token_offset,
                "state_token_offset": self._config.state_token_offset,
                "action_token_offset": self._config.action_token_offset,
                "row_mapping": "actual_row = row_start + action_token_offset + token_idx",
                "applies_to_capture_target": "action_expert",
            },
            "execution_policy": {
                "replan_steps": self._config.replan_steps,
                "observed_executed_chunk_lens": sorted(self._observed_executed_chunk_lens),
                "filter_rule": "keep token_idx < executed_chunk_len",
            },
            "flush_every_rows": self._config.flush_every_rows,
            "append_to_existing": self._config.append_to_existing,
            "global_forward_idx_next": self._global_forward_idx + 1,
            "hook_site": {
                "file": "third_party/openpi/src/openpi/models/gemma.py",
                "expression": "xs = [_gated_residual(x, y, gate) ...] after the per-expert MLP",
                "analogue_of": "OpenVLA decoder layer output / resid_post",
            },
            "paths": {
                "prompt_records": "prompt_records.jsonl",
                "trajectory_records": "trajectory_records.jsonl",
                "activation_index": str(self._activation_index_path.relative_to(self.run_dir)),
                "activations_dir": str(self._activations_dir.relative_to(self.run_dir)),
                "notes": "notes.md",
                "log": str(self._log_path.relative_to(self.run_dir)),
            },
            "sample_hidden_shapes": self._sample_hidden_shapes,
            "layer_state": {
                f"layer_{layer_idx:02d}": {
                    "next_shard_id": self._layers[layer_idx].shard_id,
                    "total_rows_written": self._layers[layer_idx].total_rows,
                }
                for layer_idx in self._selected_layers
            },
        }
        metadata_path.write_text(json.dumps(ensure_jsonable(payload), indent=2) + "\n", encoding="utf-8")

    def _restore_existing_state(self, existing_index_records: list[dict[str, Any]]) -> None:
        if not self._config.append_to_existing:
            return

        self._global_forward_idx = max(
            (int(record.get("global_forward_idx", 0)) for record in existing_index_records),
            default=0,
        )

        if existing_index_records:
            self._write_log(
                f"Resuming existing run with {len(existing_index_records)} activation-index rows; "
                f"next_global_forward_idx={self._global_forward_idx + 1}"
            )

        for layer_idx in self._selected_layers:
            layer_buffer = self._layers[layer_idx]
            pattern = f"layer_{layer_idx:02d}_shard_*.pt"
            shard_paths = sorted(self._activations_dir.glob(pattern))
            if shard_paths:
                layer_buffer.shard_id = max(self._parse_shard_id(path) for path in shard_paths) + 1
            layer_buffer.total_rows = sum(self._safe_torch_rows(path) for path in shard_paths)

        metadata_path = self.run_dir / "run_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
            sample_hidden_shapes = metadata.get("sample_hidden_shapes")
            if isinstance(sample_hidden_shapes, dict):
                self._sample_hidden_shapes = ensure_jsonable(sample_hidden_shapes)

    @staticmethod
    def _parse_shard_id(path: Path) -> int:
        match = re.search(r"_shard_(\d+)\.pt$", path.name)
        if match is None:
            return 0
        return int(match.group(1))

    @staticmethod
    def _safe_torch_rows(path: Path) -> int:
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(
                f"Expected dense 2D tensor in shard {path}, got {type(tensor)} shape={getattr(tensor, 'shape', None)}"
            )
        return int(tensor.shape[0])

    @staticmethod
    def _dedupe_run_dir(base_dir: Path) -> Path:
        if not base_dir.exists():
            return base_dir
        suffix = 1
        while True:
            candidate = base_dir.parent / f"{base_dir.name}_{suffix:02d}"
            if not candidate.exists():
                return candidate
            suffix += 1


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_run_dir(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    index_path = run_dir / "activation_index.jsonl"
    records = _load_jsonl(index_path)
    if not records:
        raise FileNotFoundError(f"No activation_index records found at {index_path}")

    checked_shards: dict[str, tuple[int, int]] = {}
    bad_entries: list[dict[str, Any]] = []
    for record in records:
        shard_rel = record["shard_path"]
        if shard_rel not in checked_shards:
            shard_tensor = torch.load(run_dir / shard_rel, map_location="cpu")
            if not isinstance(shard_tensor, torch.Tensor):
                raise TypeError(f"Shard {shard_rel} did not load as a torch.Tensor")
            if shard_tensor.ndim != 2:
                raise ValueError(f"Shard {shard_rel} expected ndim=2, got shape={tuple(shard_tensor.shape)}")
            checked_shards[shard_rel] = (int(shard_tensor.shape[0]), int(shard_tensor.shape[1]))
        num_rows, width = checked_shards[shard_rel]
        row_start = int(record["row_start"])
        row_end = int(record["row_end"])
        if not (0 <= row_start < row_end <= num_rows):
            bad_entries.append(record)
        if int(record["d_model"]) != width:
            bad_entries.append(record)

    if bad_entries:
        raise ValueError(f"Found {len(bad_entries)} invalid activation index entries.")

    prompt_records = _load_jsonl(run_dir / "prompt_records.jsonl")
    trajectory_records = _load_jsonl(run_dir / "trajectory_records.jsonl")
    return {
        "run_dir": str(run_dir),
        "activation_index_entries": len(records),
        "checked_shards": len(checked_shards),
        "prompt_records": len(prompt_records),
        "trajectory_records": len(trajectory_records),
        "sample_record": records[0],
    }


def lookup_activation_rows(
    run_dir: str | Path,
    *,
    prompt: str | None = None,
    task_description: str | None = None,
    episode_num: int | None = None,
    step_in_episode: int | None = None,
    global_forward_idx: int | None = None,
    layer_idx: int | None = None,
) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    records = _load_jsonl(run_dir / "activation_index.jsonl")

    def _matches(record: dict[str, Any]) -> bool:
        if (
            prompt is not None
            and record.get("prompt_task_description") != prompt
            and record.get("task_description") != prompt
        ):
            return False
        if task_description is not None and record.get("task_description") != task_description:
            return False
        if episode_num is not None and int(record.get("episode_num", -1)) != episode_num:
            return False
        if step_in_episode is not None and int(record.get("step_in_episode", -1)) != step_in_episode:
            return False
        if global_forward_idx is not None and int(record.get("global_forward_idx", -1)) != global_forward_idx:
            return False
        if layer_idx is not None and int(record.get("layer_idx", -1)) != layer_idx:
            return False
        return True

    return [record for record in records if _matches(record)]
