"""CLI: task-local agglomerative clustering of event features.

Reads `event_features.jsonl` produced by `scripts/build_event_features.py`,
writes `cluster_assignments.jsonl` + `clusters.jsonl` + `summary.json`.
"""

import argparse
import json
from pathlib import Path

from event_sae.events.cluster import cluster_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-local agglomerative clustering of event features.")
    parser.add_argument("--event-features-path", required=True, help="Path to event_features.jsonl")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: 'clusters/' next to event_features.jsonl)",
    )
    parser.add_argument("--vision-weight", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=0.5)
    parser.add_argument("--progress-weight", type=float, default=0.4)
    parser.add_argument("--distance-threshold", type=float, default=0.18)
    parser.add_argument(
        "--min-coverage", type=float, default=0.5,
        help="Label canonical clusters only; retain all clusters. Ranking sets its own threshold. Default: 0.5.",
    )
    parser.add_argument("--num-exemplars", type=int, default=5)
    args = parser.parse_args()

    event_features_path = Path(args.event_features_path).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else event_features_path.with_suffix("").with_name("clusters").resolve()
    )

    summary = cluster_events(
        event_features_path=event_features_path,
        output_dir=output_dir,
        vision_weight=args.vision_weight,
        state_weight=args.state_weight,
        progress_weight=args.progress_weight,
        distance_threshold=args.distance_threshold,
        min_coverage=args.min_coverage,
        num_exemplars=args.num_exemplars,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
