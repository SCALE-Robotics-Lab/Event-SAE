"""Extract sparse top-k SAE features from indexed dense activation shards."""

import argparse
import json
import hashlib
from collections import defaultdict
from pathlib import Path

import torch

from event_sae.sae import load_batch_topk_sae


def _valid_prefix(record: dict, tokenizer, token_idx: torch.Tensor) -> torch.Tensor:
    # pi05_libero: two present 256-token images, one absent image, then 200 text slots.
    if int(record.get("seq_len", 0)) != 968 or int(record.get("batch_size", 1)) != 1:
        raise ValueError("attention-valid requires pi05_libero prefixes with shape [1, 968, width].")
    prompt = record.get("task_description", record.get("prompt_task_description"))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Missing saved instruction for attention-valid extraction.")
    prompt = prompt.strip().replace("_", " ").replace("\n", " ")
    text_length = min(200, len(tokenizer.encode(prompt, add_bos=True) + tokenizer.encode("\n")))
    return (token_idx < 512) | ((token_idx >= 768) & (token_idx < 768 + text_length))


def _load_index(index_path: Path, layer_idx: int) -> dict[str, list[dict]]:
    by_shard: dict[str, list[dict]] = defaultdict(list)
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if int(record["layer_idx"]) != layer_idx:
                continue
            by_shard[str(record["shard_path"])].append(record)
    for records in by_shard.values():
        records.sort(key=lambda r: int(r["row_start"]))
    return by_shard


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline SAE encode of dense activation shards.")
    parser.add_argument(
        "--dense-dir",
        required=True,
        help=(
            "Directory that contains activation_index.jsonl; shard paths in the "
            "index are resolved relative to it. For OpenPI eval runs pass the run "
            "root (the index sits at the root). For OpenVLA collection runs pass "
            "the per-target subdir (e.g. sae_activations/resid_post)."
        ),
    )
    parser.add_argument("--sae-checkpoint", required=True, help="Path to trained ae.pt")
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--pg-mode", choices=("all", "attention-valid"), default="all")
    parser.add_argument("--tokenizer-model", type=Path, help="PaliGemma SentencePiece model for attention-valid mode.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: {dense_dir}/topk_activations).",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available else cpu).")
    args = parser.parse_args()

    dense_dir = Path(args.dense_dir).resolve()
    index_path = dense_dir / "activation_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (dense_dir / "topk_activations").resolve()
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sae, config = load_batch_topk_sae(Path(args.sae_checkpoint), device=device)
    trainer_cfg = config["trainer"]
    activation_dim = int(trainer_cfg["activation_dim"])
    dict_size = int(trainer_cfg["dict_size"])
    if not (1 <= args.topk <= dict_size):
        raise ValueError(f"topk must be in [1, {dict_size}], got {args.topk}")

    index = _load_index(index_path, layer_idx=args.layer_idx)
    if not index:
        raise RuntimeError(f"No index records found for layer {args.layer_idx} in {index_path}")

    # Probe a record for OpenPI-specific fields. ``capture_target`` and
    # ``executed_chunk_len`` propagate to the manifest so the scorer can
    # auto-detect step_mapping (action_executed for AE, chunk_executed for
    # PG, inference_step for OpenVLA legacy).
    probe_record = next(iter(index.values()))[0]
    capture_target = probe_record.get("capture_target")
    if any(r.get("capture_target") != capture_target for records in index.values() for r in records):
        raise ValueError("Extract one capture target at a time.")
    tokenizer = None
    if args.pg_mode == "attention-valid":
        if capture_target != "paligemma":
            raise ValueError("attention-valid is only supported for PaliGemma.")
        metadata = json.loads((dense_dir / "run_metadata.json").read_text())
        if metadata.get("policy_config_name") != "pi05_libero":
            raise ValueError("attention-valid recovery currently supports pi05_libero only.")
        if args.tokenizer_model is None:
            parser.error("--tokenizer-model is required for attention-valid mode")
        import sentencepiece

        tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer_model))
    output_dir.mkdir(parents=True, exist_ok=False)
    executed_chunk_len_seen = sorted({
        int(r["executed_chunk_len"])
        for records in index.values()
        for r in records
        if r.get("executed_chunk_len") is not None
    })

    manifest = {
        "format": "token_topk_sparse_v1",
        "layer": args.layer_idx,
        "sae_path": str(args.sae_checkpoint),
        "dict_size": dict_size,
        "activation_dim": activation_dim,
        "topk": args.topk,
        "capture_target": capture_target,
        "pg_mode": args.pg_mode if capture_target == "paligemma" else None,
        "executed_chunk_lens_seen": executed_chunk_len_seen,
        "num_shards": 0,
        "total_rows": 0,
        "shards": [],
    }
    if tokenizer is not None:
        manifest["tokenizer_sha256"] = hashlib.sha256(args.tokenizer_model.read_bytes()).hexdigest()

    shard_names = sorted(index.keys())
    total_rows = 0
    for src_shard_name in shard_names:
        src_shard_path = dense_dir / src_shard_name
        if not src_shard_path.is_file():
            raise FileNotFoundError(f"Missing dense shard: {src_shard_path}")
        dense = torch.load(src_shard_path, map_location="cpu", weights_only=True).to(torch.float32)
        if dense.ndim != 2 or int(dense.shape[1]) != activation_dim:
            raise ValueError(
                f"Unexpected dense shard shape {tuple(dense.shape)} in {src_shard_path}; "
                f"expected (N, {activation_dim})"
            )
        records = index[src_shard_name]
        n_rows = int(dense.shape[0])

        episode_num = torch.zeros((n_rows,), dtype=torch.int64)
        step_in_episode = torch.zeros((n_rows,), dtype=torch.int64)
        global_forward_idx = torch.zeros((n_rows,), dtype=torch.int64)
        token_idx = torch.zeros((n_rows,), dtype=torch.int64)
        batch_idx = torch.zeros((n_rows,), dtype=torch.int64)
        task_id = torch.full((n_rows,), -1, dtype=torch.int64)
        initial_state = torch.full((n_rows,), -1, dtype=torch.int64)
        task_suite_name = [None] * n_rows
        keep = torch.ones(n_rows, dtype=torch.bool)
        # OpenPI-only: per-row chunk_start_step + executed_chunk_len so
        # `score_cluster_features` can compute `_effective_steps` under the
        # chosen step_mapping. OpenVLA records leave these at -1 (sentinel).
        chunk_start_step = torch.full((n_rows,), -1, dtype=torch.int64)
        executed_chunk_len = torch.full((n_rows,), -1, dtype=torch.int64)
        previous_end = 0
        for record in records:
            r0, r1 = int(record["row_start"]), int(record["row_end"])
            if r0 != previous_end or not r0 < r1 <= n_rows:
                raise ValueError(f"Noncontiguous or invalid row interval in {src_shard_name}: {r0}:{r1}")
            previous_end = r1
            if int(record.get("batch_size", 1)) != 1 or record.get("hidden_shape", [1])[0] != 1:
                raise ValueError("Offline extraction expects single-observation rollout records.")
            task_id[r0:r1] = int(record["task_id"])
            initial_state[r0:r1] = int(record.get("initial_state", record.get("task_episode_idx", -1)))
            task_suite_name[r0:r1] = [record.get("task_suite_name")] * (r1-r0)
            episode_num[r0:r1] = int(record.get("episode_num") or 0)
            global_forward_idx[r0:r1] = int(record.get("global_forward_idx") or 0)
            # Per-token env-step mapping. OpenPI records each forward as one
            # chunked inference covering `seq_len` future tokens; the env step
            # a token corresponds to is `chunk_start + token_idx`. OpenVLA
            # records each forward as one env-step (no chunking), and all
            # rows of a record share the same `step_in_episode`.
            tokens_local = torch.arange(r1 - r0, dtype=torch.int64)
            token_idx[r0:r1] = tokens_local
            if tokenizer is not None:
                if r1 - r0 != int(record["seq_len"]):
                    raise ValueError("Prefix row count does not match seq_len.")
                keep[r0:r1] = _valid_prefix(record, tokenizer, tokens_local)
            chunk_start = record.get("chunk_start_step")
            if chunk_start is None:
                chunk_start = record.get("action_chunk_start_step")
            if chunk_start is not None:
                chunk_start_step[r0:r1] = int(chunk_start)
                step_in_episode[r0:r1] = int(chunk_start) + tokens_local
            else:
                step_in_episode[r0:r1] = int(record.get("step_in_episode") or 0)
            if record.get("executed_chunk_len") is not None:
                executed_chunk_len[r0:r1] = int(record["executed_chunk_len"])
            elif capture_target in ("paligemma", "action_expert"):
                raise ValueError("OpenPI index is missing executed_chunk_len; use a collection with planned chunk metadata.")
        if previous_end != n_rows:
            raise ValueError(f"Index does not cover every row in {src_shard_name}.")

        with torch.no_grad():
            encoded = sae.encode(dense[keep].to(device))
            values, indices = torch.topk(encoded, k=args.topk, dim=-1)
            values = values.float().cpu()
            indices = indices.to(torch.int32).cpu()

        out_shard_name = f"shard_{manifest['num_shards']:06d}.pt"
        torch.save(
            {
                "episode_num": episode_num[keep],
                "step_in_episode": step_in_episode[keep],
                "global_forward_idx": global_forward_idx[keep],
                "batch_idx": batch_idx[keep],
                "token_idx": token_idx[keep],
                "task_id": task_id[keep],
                "initial_state": initial_state[keep],
                "task_suite_name": [name for name, selected in zip(task_suite_name, keep.tolist()) if selected],
                "chunk_start_step": chunk_start_step[keep],
                "executed_chunk_len": executed_chunk_len[keep],
                "top_feature_ids": indices,
                "top_feature_vals": values,
            },
            output_dir / out_shard_name,
        )
        n_rows = int(keep.sum())
        manifest["shards"].append(
            {
                "shard_idx": manifest["num_shards"],
                "path": out_shard_name,
                "num_rows": n_rows,
                "row_start": total_rows,
                "row_end": total_rows + n_rows,
            }
        )
        manifest["num_shards"] += 1
        total_rows += n_rows
        print(f"Encoded {src_shard_name} → {out_shard_name}  rows={n_rows}")

    manifest["total_rows"] = total_rows
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {output_dir / 'manifest.json'}")
    print(f"Total rows: {total_rows}")


if __name__ == "__main__":
    main()
