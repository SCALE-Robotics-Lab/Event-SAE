"""Object poses and maximum lift from LIBERO rollout observations."""


def object_poses_from_observation(obs):
    poses = {}
    for key, position in obs.items():
        if not key.endswith("_pos") or key.startswith("robot") or "_to_robot" in key:
            continue
        name = key[:-len("_pos")]
        if f"{name}_quat" not in obs:
            continue
        poses[name] = {
            "position": [float(value) for value in position],
            "quaternion": [float(value) for value in obs[f"{name}_quat"]],
        }
    return poses


def maximum_lift(positions):
    """Maximum height increase in meters relative to the first recorded position."""
    reference_height = positions[0][2]
    return max(position[2] - reference_height for position in positions)
