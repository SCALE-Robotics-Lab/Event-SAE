"""CLI: build event-feature score matrix.

Joins `token_topk_sparse_v1` shards from `scripts/extract_topk.py` with event clusters, and
writes a single `.pt` payload with the `(num_clusters, dict_size)` score
matrix and per-row top-N feature summaries.
"""

import argparse
import json
from pathlib import Path

from event_sae.scoring import score_cluster_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-feature score matrix.")
    parser.add_argument(
        "--topk-run-dir",
        required=True,
        help="Directory containing manifest.json + token_topk_sparse_v1 shards "
        "(either an online EVAL run dir or the offline extract_topk output dir).",
    )
    parser.add_argument("--event-features-path", required=True, help="Path to event_features.jsonl")
    parser.add_argument("--cluster-assignments-path", required=True, help="Path to cluster_assignments.jsonl")
    clusters = parser.add_mutually_exclusive_group(required=True)
    clusters.add_argument("--clusters-path", help="Path to clusters.jsonl (no annotations needed)")
    clusters.add_argument("--cluster-annotations-path", help="Path to legacy cluster_annotations.jsonl")
    parser.add_argument(
        "--prompt-records-path",
        default=None,
        help=(
            "Episode/task mapping for all cached episodes, including those without events. "
            "If omitted, task means include only episodes that produced events."
        ),
    )
    parser.add_argument("--output-path", required=True, help="Where to save the score matrix .pt payload")
    parser.add_argument("--window-size", type=int, default=5, help="Half-window size in env steps (default: 5)")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N features to summarize per row (default: 20)")
    parser.add_argument(
        "--step-mapping",
        choices=("auto", "action_executed", "chunk_executed", "inference_step"),
        default="auto",
        help=(
            "How shard rows map to env timesteps. action_executed uses chunk_start + token_idx "
            "(OpenPI AE default); chunk_executed broadcasts each row to all executed env steps "
            "of its chunk (OpenPI PG default); inference_step uses step_in_episode directly "
            "(OpenVLA legacy). 'auto' picks per manifest.capture_target."
        ),
    )
    args = parser.parse_args()

    summary = score_cluster_features(
        topk_run_dir=Path(args.topk_run_dir),
        event_features_path=Path(args.event_features_path),
        cluster_assignments_path=Path(args.cluster_assignments_path),
        cluster_annotations_path=Path(args.cluster_annotations_path) if args.cluster_annotations_path else None,
        clusters_path=Path(args.clusters_path) if args.clusters_path else None,
        output_path=Path(args.output_path),
        window_size=args.window_size,
        top_n=args.top_n,
        step_mapping=args.step_mapping,
        prompt_records_path=Path(args.prompt_records_path) if args.prompt_records_path else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
