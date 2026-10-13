"""CLI: extract AWE keyframes from a `trajectory_records.jsonl` file.

Writes one summary JSON (`waypoint_summary.json`) under `--output-dir`,
listing per-episode waypoint indices.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from event_sae.keyframes import (
    extract_waypoints_dp,
    filter_episodes,
    gripper_toggle_indices,
    load_episode_trajectories,
    require_geometric_gripper_inputs,
)


def _default_output_dir(records_path: Path, err_threshold: float, waypoint_mode: str) -> Path:
    threshold_tag = f"{err_threshold:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return records_path.parent / "keyframes" / f"dp_{waypoint_mode}_err{threshold_tag}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract AWE keyframes from trajectory_records.jsonl.")
    parser.add_argument("--trajectory-records-path", required=True, help="Path to trajectory_records.jsonl")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: derived from records path)")
    parser.add_argument(
        "--waypoint-mode",
        choices=("pos_only", "geometric_gripper"),
        default="pos_only",
        help="'geometric_gripper' requires eef_quat + gripper_action in records",
    )
    parser.add_argument("--err-threshold", type=float, default=0.05, help="AWE error threshold")
    parser.add_argument(
        "--success-filter", choices=("all", "success", "failure"), default="all"
    )
    parser.add_argument("--task-description", default=None, help="Exact task_description filter")
    parser.add_argument("--suite", default=None, help="Suite for legacy trajectories that omit task_suite_name")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--show-awe-logs", action="store_true", help="Don't suppress AWE internal prints")
    args = parser.parse_args()

    records_path = Path(args.trajectory_records_path).resolve()
    if not records_path.is_file():
        raise FileNotFoundError(f"trajectory_records.jsonl not found: {records_path}")

    episodes = load_episode_trajectories(records_path)
    if args.suite is not None:
        for episode in episodes:
            if episode.task_suite_name not in (None, args.suite):
                raise ValueError("--suite conflicts with the trajectory's task_suite_name")
            episode.task_suite_name = args.suite
    selected = filter_episodes(
        episodes,
        success_filter=args.success_filter,
        task_description=args.task_description,
        max_episodes=args.max_episodes,
    )
    if not selected:
        raise ValueError("No episodes matched the requested filters.")
    if args.waypoint_mode == "geometric_gripper":
        require_geometric_gripper_inputs(selected)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else _default_output_dir(records_path, args.err_threshold, args.waypoint_mode).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_trajectory_records_path": str(records_path),
        "output_dir": str(output_dir),
        "waypoint_mode": args.waypoint_mode,
        "err_threshold": float(args.err_threshold),
        "success_filter": args.success_filter,
        "task_description_filter": args.task_description,
        "num_available_episodes": len(episodes),
        "num_selected_episodes": len(selected),
        "episodes": [],
    }

    extraction_times: list[float] = []
    wall_start = time.perf_counter()
    for idx, episode in enumerate(selected):
        if len(episode.positions) < 2:
            continue
        ep_start = time.perf_counter()
        waypoints = extract_waypoints_dp(
            episode,
            waypoint_mode=args.waypoint_mode,
            err_threshold=args.err_threshold,
            show_awe_logs=args.show_awe_logs,
        )
        ep_seconds = time.perf_counter() - ep_start
        extraction_times.append(ep_seconds)
        summary["episodes"].append(
            {
                "episode_num": episode.episode_num,
                "task_id": episode.task_id,
                "task_suite_name": episode.task_suite_name,
                "task_episode_idx": episode.task_episode_idx,
                "task_description": episode.task_description,
                "prompt_task_description": episode.prompt_task_description,
                "success": episode.success,
                "num_steps": int(len(episode.positions)),
                "waypoint_indices": waypoints,
                "num_waypoints": len(waypoints),
                "waypoint_positions": episode.positions[waypoints].tolist(),
                "waypoint_mode": args.waypoint_mode,
                "has_eef_quat": episode.quaternions is not None,
                "has_gripper_action": episode.gripper_actions is not None,
                "gripper_toggle_indices": gripper_toggle_indices(episode),
                "extraction_seconds": ep_seconds,
            }
        )
        mean_seconds = float(np.mean(extraction_times))
        eta = mean_seconds * (len(selected) - (idx + 1))
        print(
            f"[{idx + 1}/{len(selected)}] episode={episode.episode_num} "
            f"task={episode.task_id} steps={len(episode.positions)} "
            f"waypoints={len(waypoints)} time={ep_seconds:.2f}s eta={eta:.1f}s"
        )

    total = time.perf_counter() - wall_start
    summary["total_extraction_seconds"] = total
    summary["mean_episode_extraction_seconds"] = (
        float(np.mean(extraction_times)) if extraction_times else 0.0
    )

    summary_path = output_dir / "waypoint_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")
    print(f"Total time: {total:.2f}s; mean per episode: {summary['mean_episode_extraction_seconds']:.2f}s")


if __name__ == "__main__":
    main()
