"""Geometric trajectory errors used by AWE."""
import numpy as np
from scipy.spatial.transform import Rotation
import robosuite.utils.transform_utils as T


def linear_interpolation(p1, p2, t):
    """Compute the linear interpolation between two 3D points"""
    return p1 + t * (p2 - p1)


def point_line_distance(point, line_start, line_end):
    """Compute the shortest distance between a 3D point and a line segment defined by two 3D points"""
    line_vector = line_end - line_start
    point_vector = point - line_start
    # t represents the position of the orthogonal projection of the given point onto the infinite line defined by the segment
    t = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
    t = np.clip(t, 0, 1)
    projection = linear_interpolation(line_start, line_end, t)
    return np.linalg.norm(point - projection)


def point_quat_distance(point, quat_start, quat_end, t, total):
    pred_point = T.quat_slerp(quat_start, quat_end, fraction=t / total)
    err_quat = (
        Rotation.from_quat(pred_point) * Rotation.from_quat(point).inv()
    ).magnitude()
    return err_quat


def geometric_waypoint_trajectory(actions, gt_states, waypoints, return_list=False):
    """Compute the geometric trajectory from the waypoints"""

    # prepend 0 to the waypoints for geometric computation
    if waypoints[0] != 0:
        waypoints = [0] + waypoints
    gt_pos = [p["robot0_eef_pos"] for p in gt_states]
    gt_quat = [p["robot0_eef_quat"] for p in gt_states]

    keypoints_pos = [actions[k, :3] for k in waypoints]
    keypoints_quat = [gt_quat[k] for k in waypoints]

    state_err = []

    n_segments = len(waypoints) - 1

    for i in range(n_segments):
        # Get the current keypoint and the next keypoint
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]
        start_keypoint_quat = keypoints_quat[i]
        end_keypoint_quat = keypoints_quat[i + 1]

        # Select ground truth points within the current segment
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_pos[start_idx:end_idx]
        segment_points_quat = gt_quat[start_idx:end_idx]

        # Compute the shortest distances between ground truth points and the current segment
        for i in range(len(segment_points_pos)):
            pos_err = point_line_distance(
                segment_points_pos[i], start_keypoint_pos, end_keypoint_pos
            )
            rot_err = point_quat_distance(
                segment_points_quat[i],
                start_keypoint_quat,
                end_keypoint_quat,
                i,
                len(segment_points_quat),
            )
            state_err.append(pos_err + rot_err)

    # print the average and max error for pos and rot
    # print(f"Average pos error: {np.mean(pos_err_list):.6f} \t Average rot error: {np.mean(rot_err_list):.6f}")
    # print(f"Max pos error: {np.max(pos_err_list):.6f} \t Max rot error: {np.max(rot_err_list):.6f}")

    if return_list:
        return total_traj_err(state_err), state_err
    return total_traj_err(state_err)


def pos_only_geometric_waypoint_trajectory(
    actions, gt_states, waypoints, return_list=False
):
    """Compute the geometric trajectory from the waypoints"""

    # prepend 0 to the waypoints for geometric computation
    if waypoints[0] != 0:
        waypoints = [0] + waypoints

    keypoints_pos = [actions[k] for k in waypoints]
    state_err = []
    n_segments = len(waypoints) - 1

    for i in range(n_segments):
        # Get the current keypoint and the next keypoint
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]

        # Select ground truth points within the current segment
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_states[start_idx:end_idx]

        # Compute the shortest distances between ground truth points and the current segment
        for i in range(len(segment_points_pos)):
            pos_err = point_line_distance(
                segment_points_pos[i], start_keypoint_pos, end_keypoint_pos
            )
            state_err.append(pos_err)

    # print the average and max error
    print(
        f"Average pos error: {np.mean(state_err):.6f} \t Max pos error: {np.max(state_err):.6f}"
    )

    if return_list:
        return total_traj_err(state_err), state_err
    else:
        return total_traj_err(state_err)


def total_traj_err(err_list):
    # return np.mean(err_list)
    return np.max(err_list)
