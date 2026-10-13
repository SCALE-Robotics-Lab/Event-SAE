"""AWE keyframe extraction from LIBERO `trajectory_records.jsonl`.

Backbone-agnostic — operates on per-step `eef_pos` / `eef_quat` /
`gripper_action` records produced by any rollout (openVLA, openpi, etc.).

The dynamic-programming core call is `waypoint_extraction.dp_waypoint_selection`
(see `third_party/awe/`).
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from waypoint_extraction import dp_waypoint_selection


@dataclass
class EpisodeTrajectory:
    episode_num: int
    task_id: int
    task_episode_idx: int
    task_description: str
    prompt_task_description: str
    success: bool
    step_indices: list[int]
    positions: np.ndarray
    quaternions: np.ndarray | None = None
    gripper_actions: np.ndarray | None = None
    gripper_qpos: np.ndarray | None = None
    task_suite_name: str | None = None


def _load_optional_vector_field(records: list[dict], key: str) -> np.ndarray | None:
    if not all(key in record for record in records):
        return None
    return np.asarray([record[key] for record in records], dtype=np.float32)


def _load_optional_scalar_field(records: list[dict], key: str) -> np.ndarray | None:
    if not all(key in record for record in records):
        return None
    return np.asarray([float(record[key]) for record in records], dtype=np.float32)


def load_episode_trajectories(path: Path) -> list[EpisodeTrajectory]:
    """Load and group `trajectory_records.jsonl` into per-episode `EpisodeTrajectory`."""
    grouped: dict[tuple[int, int, int], list[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            key = (
                int(record["episode_num"]),
                int(record["task_id"]),
                int(record["task_episode_idx"]),
            )
            grouped.setdefault(key, []).append(record)

    episodes: list[EpisodeTrajectory] = []
    for (episode_num, task_id, task_episode_idx), records in sorted(grouped.items()):
        records.sort(key=lambda item: int(item["step_in_episode"]))
        first = records[0]
        positions = np.asarray([record["eef_pos"] for record in records], dtype=np.float32)
        quaternions = _load_optional_vector_field(records, "eef_quat")
        gripper_actions = _load_optional_scalar_field(records, "gripper_action")
        gripper_qpos = _load_optional_vector_field(records, "gripper_qpos")
        step_indices = [int(record["step_in_episode"]) for record in records]
        episodes.append(
            EpisodeTrajectory(
                episode_num=episode_num,
                task_id=task_id,
                task_episode_idx=task_episode_idx,
                task_description=str(first.get("task_description", "")),
                prompt_task_description=str(first.get("prompt_task_description", first.get("task_description", ""))),
                success=bool(records[-1]["done"]),
                step_indices=step_indices,
                positions=positions,
                quaternions=quaternions,
                gripper_actions=gripper_actions,
                gripper_qpos=gripper_qpos,
                task_suite_name=first.get("task_suite_name"),
            )
        )
    return episodes


def filter_episodes(
    episodes: list[EpisodeTrajectory],
    *,
    success_filter: str = "all",
    task_description: str | None = None,
    max_episodes: int | None = None,
) -> list[EpisodeTrajectory]:
    """Filter episodes by final-done flag, task description, and count."""
    filtered = episodes
    if success_filter == "success":
        filtered = [episode for episode in filtered if episode.success]
    elif success_filter == "failure":
        filtered = [episode for episode in filtered if not episode.success]
    if task_description is not None:
        filtered = [episode for episode in filtered if episode.task_description == task_description]
    if max_episodes is not None:
        filtered = filtered[:max_episodes]
    return filtered


def require_geometric_gripper_inputs(episodes: list[EpisodeTrajectory]) -> None:
    missing_quat = [e.episode_num for e in episodes if e.quaternions is None]
    missing_gripper = [e.episode_num for e in episodes if e.gripper_actions is None]
    if missing_quat or missing_gripper:
        raise ValueError(
            "waypoint_mode='geometric_gripper' requires every selected episode to have "
            f"eef_quat and gripper_action. Missing eef_quat episodes: {missing_quat[:10]}; "
            f"missing gripper_action episodes: {missing_gripper[:10]}"
        )


def gripper_toggle_indices(episode: EpisodeTrajectory) -> list[int]:
    """Step indices where the gripper command flipped sign."""
    if episode.gripper_actions is None:
        return []
    return [
        int(episode.step_indices[idx])
        for idx in range(len(episode.gripper_actions) - 1)
        if episode.gripper_actions[idx] != episode.gripper_actions[idx + 1]
    ]


def extract_waypoints_dp(
    episode: EpisodeTrajectory,
    *,
    waypoint_mode: str = "pos_only",
    err_threshold: float = 0.05,
    show_awe_logs: bool = False,
) -> list[int]:
    """Run AWE `dp_waypoint_selection` on a single episode and return waypoint step indices.

    `waypoint_mode='pos_only'` uses end-effector position only; `'geometric_gripper'`
    additionally consumes `eef_quat` and `gripper_action`.
    """
    if waypoint_mode == "pos_only":
        call_kwargs = {
            "env": None,
            "actions": episode.positions,
            "gt_states": episode.positions,
            "err_threshold": err_threshold,
            "pos_only": True,
        }
    elif waypoint_mode == "geometric_gripper":
        if episode.quaternions is None or episode.gripper_actions is None:
            raise ValueError(
                "geometric_gripper mode requires both eef_quat and gripper_action per step."
            )
        call_kwargs = {
            "env": None,
            "actions": np.concatenate(
                [episode.positions, episode.gripper_actions[:, np.newaxis]],
                axis=1,
            ),
            "gt_states": [
                {
                    "robot0_eef_pos": episode.positions[idx],
                    "robot0_eef_quat": episode.quaternions[idx],
                }
                for idx in range(len(episode.positions))
            ],
            "err_threshold": err_threshold,
            "pos_only": False,
        }
    else:
        raise ValueError(f"Unsupported waypoint mode: {waypoint_mode}")

    if show_awe_logs:
        waypoints = dp_waypoint_selection(**call_kwargs)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            waypoints = dp_waypoint_selection(**call_kwargs)
    return [int(idx) for idx in waypoints]
