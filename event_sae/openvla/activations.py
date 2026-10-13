"""Dense OpenVLA decoder-layer activations and their rollout row indices."""


from __future__ import annotations

import json
from pathlib import Path
from typing import List
import torch


class ActivationCollectHandle:
    """Handle returned by collection helpers. Call `.remove()` at end of
    rollout to flush the last partial shard and unregister all hooks.
    """

    def __init__(self, hooks: List[object], flush_fn):
        self._hooks = hooks
        self._flush_fn = flush_fn

    def remove(self) -> None:
        self._flush_fn()
        for h in self._hooks:
            h.remove()


def apply_collect_hooks(
    model,
    layer_idxs: List[int],
    output_dir: str | Path,
    flush_every: int = 10_000,
) -> ActivationCollectHandle:
    """Hook each `i` in `layer_idxs` on `model.language_model.model.layers[i]`
    and write dense shards + `activation_index.jsonl` into `output_dir`.

    Buffer flushes to a new shard once it reaches `flush_every` rows
    (forward boundaries are never split across shards).
    """
    decoder_layers = model.language_model.model.layers
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "activation_index.jsonl"
    index_file = index_path.open("w", encoding="utf-8")

    buffers = {idx: [] for idx in layer_idxs}
    pending_records = {idx: [] for idx in layer_idxs}
    buffered = {idx: 0 for idx in layer_idxs}
    shard_ids = {idx: 0 for idx in layer_idxs}
    global_forward_idx = {"value": 0}

    def flush_layer(idx: int) -> None:
        if not buffers[idx]:
            return
        acts = torch.cat(buffers[idx], dim=0)
        shard_name = f"layer_{idx:02d}_shard_{shard_ids[idx]:06d}.pt"
        torch.save(acts, output_dir / shard_name)
        for record in pending_records[idx]:
            record["shard_path"] = shard_name
            index_file.write(json.dumps(record) + "\n")
        index_file.flush()
        shard_ids[idx] += 1
        buffers[idx].clear()
        pending_records[idx].clear()
        buffered[idx] = 0

    def flush_all() -> None:
        for idx in layer_idxs:
            flush_layer(idx)
        index_file.close()

    def make_hook(idx: int):
        def hook_fn(module, inputs, output):
            hidden = output[0]
            sample = hidden.reshape(-1, hidden.shape[-1]).detach().to(torch.float32).cpu()
            n_rows = int(sample.shape[0])

            ctx = getattr(model, "_sae_hook_context", {}) or {}
            if idx == layer_idxs[0]:
                global_forward_idx["value"] += 1
            pending_records[idx].append(
                {
                    "layer_idx": int(idx),
                    "row_start": int(buffered[idx]),
                    "row_end": int(buffered[idx] + n_rows),
                    "episode_num": ctx.get("episode_num"),
                    "step_in_episode": ctx.get("step_in_episode"),
                    "task_id": ctx.get("task_id"),
                    "task_episode_idx": ctx.get("task_episode_idx"),
                    "task_suite_name": ctx.get("task_suite_name"),
                    "initial_state": ctx.get("initial_state"),
                    "hidden_shape": list(hidden.shape),
                    "token_selection": "all_forward_positions",
                    "global_forward_idx": int(global_forward_idx["value"]),
                }
            )
            buffers[idx].append(sample)
            buffered[idx] += n_rows
            if buffered[idx] >= flush_every:
                flush_layer(idx)

        return hook_fn

    hooks = [decoder_layers[idx].register_forward_hook(make_hook(idx)) for idx in layer_idxs]
    return ActivationCollectHandle(hooks=hooks, flush_fn=flush_all)
