"""CLI: render 5-frame bundle (PNGs + clip MP4) around each AWE waypoint.

Reads `waypoint_summary.json` from `scripts/extract_keyframes.py` and the
corresponding LIBERO rollout MP4s, writes `samples.jsonl` ready for
`scripts/build_event_features.py`.
"""

import argparse
from pathlib import Path

from event_sae.events.extract_media import extract_keyframe_media


def _default_output_dir(waypoint_summary_path: Path) -> Path:
    return waypoint_summary_path.parent / "samples_5frames_stride2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 5-frame bundles around each AWE waypoint."
    )
    parser.add_argument("--waypoint-summary-path", required=True, help="Path to waypoint_summary.json")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: derived from summary path)")
    parser.add_argument(
        "--frame-offsets",
        type=int,
        nargs="+",
        default=[-4, -2, 0, 2, 4],
        help="Frame offsets around each waypoint index. Default: -4 -2 0 2 4 (5 frames, stride 2).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Cap on the number of extracted samples")
    args = parser.parse_args()

    waypoint_summary_path = Path(args.waypoint_summary_path).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else _default_output_dir(waypoint_summary_path).resolve()
    )

    report = extract_keyframe_media(
        waypoint_summary_path=waypoint_summary_path,
        output_dir=output_dir,
        frame_offsets=list(args.frame_offsets),
        max_samples=args.max_samples,
    )
    print(f"Waypoints source: {report['waypoint_summary_path']}")
    print(f"Output dir: {report['output_dir']}")
    print(f"Saved samples: {report['num_samples']}")
    print(f"Skipped samples: {report['num_skipped']}")
    print(f"Samples manifest: {report['samples_path']}")


if __name__ == "__main__":
    main()
