"""Run LIBERO episodes with OpenVLA or an OpenPI policy server."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def prepare_libero(root: Path, output: Path) -> None:
    import yaml

    package = root.resolve() / "libero" / "libero"
    if not (package / "bddl_files").is_dir():
        raise ValueError(f"Not a LIBERO checkout: {root}")
    config_dir = output / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())
    os.environ.setdefault("MUJOCO_GL", "egl")
    paths = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "datasets": str(root.resolve() / "libero" / "datasets"),
        "assets": str(package / "assets"),
    }
    (config_dir / "config.yaml").write_text(yaml.safe_dump(paths))


class OpenVLA:
    def __init__(self, args):
        import tensorflow as tf
        from event_sae.openvla.eval.model import get_processor, load_model
        from event_sae.openvla.eval.utils import set_seed_everywhere

        tf.config.set_visible_devices([], "GPU")
        set_seed_everywhere(args.seed)
        self.cfg = SimpleNamespace(model=SimpleNamespace(
            checkpoint=args.checkpoint, revision=args.revision, code_revision=args.code_revision,
            attn_implementation=args.attention, center_crop=True,
        ))
        self.model = load_model(self.cfg)
        self.processor = get_processor(self.cfg)
        self.unnorm_key = args.suite
        if self.unnorm_key not in self.model.norm_stats:
            self.unnorm_key += "_no_noops"
        if self.unnorm_key not in self.model.norm_stats:
            raise ValueError(f"Missing normalization statistics for {args.suite}")
        self.metadata = {
            "checkpoint": args.checkpoint, "revision": args.revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "code_revision": args.code_revision, "attention": args.attention,
            "unnorm_key": self.unnorm_key, "center_crop": True,
        }
        self.collect_handle = None
        self.steer_handle = None
        if args.steering is not None:
            from event_sae.openvla.intervene import apply_sae_hook

            self.steer_handle = apply_sae_hook(self.model, args.steering)
            self.metadata["steering"] = args.steering
        if args.collection:
            from event_sae.openvla.activations import apply_collect_hooks

            self.collect_handle = apply_collect_hooks(
                self.model, args.collection_layers,
                args.output / "sae_activations/resid_post",
                args.collection["flush_every_rows"],
            )

    def finish_collection(self, keep):
        self.collect_handle.remove()

    def action(self, obs, prompt):
        import torch
        from event_sae.openvla.eval.libero_utils import get_libero_image
        from event_sae.openvla.eval.model import get_action
        from event_sae.openvla.eval.utils import invert_gripper_action, normalize_gripper_action

        with torch.inference_mode():
            if self.collect_handle is not None or self.steer_handle is not None:
                self.model._sae_hook_context = self.context
            action = get_action(self.model, self.processor, self.cfg,
                                {"full_image": get_libero_image(obs, 224)}, prompt, self.unnorm_key)
        return invert_gripper_action(normalize_gripper_action(action.copy()))


class OpenPI:
    def __init__(self, args):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(args.host, args.port)
        self.metadata = self.client.get_server_metadata()
        if not args.steered and any(self.metadata.get(key) for key in ("sae_reconstruction", "sae_decoder_vector_add")):
            raise ValueError("Native rollout requires a server without SAE edits.")
        self.collect = bool(args.collection)
        server_collection = self.metadata.get("sae_collection")
        if bool(server_collection) != self.collect:
            raise ValueError("Client and server must both enable collection, or both disable it.")
        if self.collect:
            if (server_collection["capture_target"] != args.collection["capture_target"]
                    or sorted(server_collection["layer_indices"]) != args.collection_layers):
                raise ValueError("Client/server collection targets or layers differ.")
            if (args.replan_steps != args.collection["replan_steps"]
                    or args.replan_steps != server_collection["replan_steps"]):
                raise ValueError("Client/server replan steps must match the collection config.")
        self.replan_steps = args.replan_steps
        self.actions = deque()
        self.chunk_records = []

    def finish_collection(self, keep):
        response = self.client.infer({"_sae_collection_control": {
            "command": "finalize_episode", "episode_num": self.context["episode_num"], "keep": keep,
        }})
        if keep and response.get("sae_collection_control", {}).get("status") != "ok":
            raise RuntimeError(f"Collection finalization failed: {response}")

    def action(self, obs, prompt):
        import numpy as np
        from openpi_client import image_tools

        if not self.actions:
            quat = np.array(obs["robot0_eef_quat"], copy=True)
            quat[3] = np.clip(quat[3], -1.0, 1.0)
            denominator = np.sqrt(1.0 - quat[3] ** 2)
            axis_angle = (np.zeros(3) if math.isclose(float(denominator), 0.0) else
                          quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator)
            state = np.concatenate((obs["robot0_eef_pos"], axis_angle, obs["robot0_gripper_qpos"]))
            def image(key):
                return image_tools.convert_to_uint8(image_tools.resize_with_pad(
                    np.ascontiguousarray(obs[key][::-1, ::-1]), 224, 224))
            request = {
                "observation/image": image("agentview_image"),
                "observation/wrist_image": image("robot0_eye_in_hand_image"),
                "observation/state": state, "prompt": prompt,
            }
            if self.collect:
                request["_sae_collection_context"] = {
                    **self.context, "chunk_start_step": self.context["step_in_episode"],
                    "action_chunk_start_step": self.context["step_in_episode"],
                    # Planned length for the index; client chunk records count actual steps.
                    "executed_chunk_len": self.replan_steps,
                    "executed_token_start": 0, "executed_token_end": self.replan_steps,
                    "execution_policy": "prefix_replan", "state": state.tolist(),
                }
            response = self.client.infer(request)
            actions = np.asarray(response["actions"])
            if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < self.replan_steps:
                raise ValueError(f"Unexpected action chunk: {actions.shape}")
            self.actions.extend(actions[:self.replan_steps])
            if self.collect:
                self.chunk_records.append({
                    **self.context, "action_chunk": actions.tolist(),
                    "action_chunk_len": len(actions), "executed_chunk_len": 0,
                })
        if self.collect:
            self.chunk_records[-1]["executed_chunk_len"] += 1
        return self.actions.popleft()


def parse_ids(value):
    """Parse comma-separated IDs and inclusive ranges, e.g. 0-9 or 0,2,5-7."""
    ids = []
    try:
        for part in value.split(","):
            bounds = [int(item) for item in part.split("-")]
            if len(bounds) == 1:
                ids.append(bounds[0])
            elif len(bounds) == 2 and bounds[0] <= bounds[1]:
                ids.extend(range(bounds[0], bounds[1] + 1))
            else:
                raise ValueError
        if min(ids) < 0 or len(set(ids)) != len(ids):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("Use distinct nonnegative IDs or ranges, e.g. 0,2,5-7") from None
    return ids


def gather_episode(output, episode_dir, result):
    for name in ("trajectory_records.jsonl", "prompt_records.jsonl", "chunk_records.jsonl"):
        source = episode_dir / name
        if source.is_file():
            with source.open() as src, (output / name).open("a") as dst:
                shutil.copyfileobj(src, dst)
    settings = result["settings"]
    episode_num = settings["episode_num"]
    video_name = f"episode_{episode_num:05d}_task_{settings['task_id']}.mp4"
    (output / "videos" / video_name).symlink_to(
        os.path.relpath(episode_dir / "rollout.mp4", output / "videos")
    )
    if settings["backend"] == "openvla" and settings["collect_config"] is not None:
        source = episode_dir / "sae_activations/resid_post"
        dense = output / "sae_activations/resid_post"
        dense.mkdir(parents=True, exist_ok=True)
        names = {}
        for shard in sorted(source.glob("layer_*_shard_*.pt")):
            name = f"{shard.stem}_episode_{episode_num:05d}.pt"
            (dense / name).symlink_to(os.path.relpath(shard, dense))
            names[shard.name] = name
        with (source / "activation_index.jsonl").open() as src, (dense / "activation_index.jsonl").open("a") as dst:
            for line in src:
                record = json.loads(line)
                record["shard_path"] = names[record["shard_path"]]
                dst.write(json.dumps(record) + "\n")


def run_batch(args, steered):
    tasks = args.task_ids if args.task_ids is not None else [args.task_id]
    states = args.initial_states if args.initial_states is not None else [args.initial_state]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "videos").mkdir()
    script = "rollout_steered.py" if steered else "rollout_native.py"
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / script)]
    varying = {"task_ids", "initial_states", "task_id", "initial_state", "episode_num", "output"}
    for key, value in vars(args).items():
        if key not in varying and value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    results = []
    for task in tasks:
        for state in states:
            episode_num = args.episode_num + len(results)
            episode_dir = args.output / "episodes" / f"task_{task}_state_{state}"
            # Separate processes preserve the single-episode policy and RNG lifecycle.
            subprocess.run(command + [
                "--task-id", str(task), "--initial-state", str(state),
                "--episode-num", str(episode_num), "--output", str(episode_dir),
            ], check=True)
            result = json.loads((episode_dir / "result.json").read_text())
            gather_episode(args.output, episode_dir, result)
            results.append({"task_id": task, "initial_state": state, "episode_num": episode_num,
                            "success": result["success"], "result_path": str(episode_dir.relative_to(args.output) / "result.json")})
    successes = sum(result["success"] for result in results)
    summary = {"num_episodes": len(results), "num_successes": successes,
               "success_rate": successes / len(results), "episodes": results}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Success: {successes}/{len(results)} = {summary['success_rate']:.1%}")


def parser(steered=False):
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--backend", choices=("openvla", "openpi"), required=True)
    result.add_argument("--libero-root", type=Path, default=Path("external/LIBERO"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--suite", choices=tuple(MAX_STEPS))
    tasks = result.add_mutually_exclusive_group()
    tasks.add_argument("--task-id", type=int, default="0")
    tasks.add_argument("--task-ids", type=parse_ids)
    states = result.add_mutually_exclusive_group()
    states.add_argument("--initial-state", type=int, default="0")
    states.add_argument("--initial-states", type=parse_ids)
    # In batch mode, episode numbering starts here and follows task/state order.
    result.add_argument("--episode-num", type=int, default=0)
    result.add_argument("--collect-config", type=Path)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--max-steps", type=int)
    model = result.add_mutually_exclusive_group()
    model.add_argument("--checkpoint")
    model.add_argument("--model-config", type=Path)
    # Explicit CLI revisions override the model YAML.
    result.add_argument("--revision")
    result.add_argument("--code-revision")
    result.add_argument("--attention", choices=("sdpa", "flash_attention_2", "eager"), default="flash_attention_2")
    result.add_argument("--host", default="localhost")
    result.add_argument("--port", type=int, default=8000)
    result.add_argument("--replan-steps", type=int, default=5)
    result.add_argument("--target-object", help="LIBERO object name for the maximum-lift readout")
    if steered:
        result.add_argument("--steer-config", type=Path, help="OpenVLA intervention YAML; configure OpenPI on its server")
    return result


def main(steered=False):
    cli = parser(steered=steered)
    args = cli.parse_args()
    if args.task_ids is not None or args.initial_states is not None:
        run_batch(args, steered)
        return
    args.steered = steered
    args.steering = None
    if steered and args.backend == "openvla":
        if not args.steer_config:
            cli.error("--steer-config is required for OpenVLA steering")
        from event_sae.steering import load_steering_config

        args.steering = load_steering_config(args.steer_config)
    if steered and args.backend == "openpi" and args.steer_config:
        cli.error("Pass --steer-config to the OpenPI server")
    args.collection = None
    if args.collect_config:
        import yaml

        with args.collect_config.open() as stream:
            args.collection = yaml.safe_load(stream)
        args.collection_layers = sorted(set(int(i) for i in str(args.collection["layer_indices"]).split(",")))
        if not args.collection_layers or min(args.collection_layers) < 0 or args.collection["flush_every_rows"] < 1:
            cli.error("Collection layers must be nonnegative and flush_every_rows positive")
    if args.model_config:
        import yaml

        with args.model_config.open() as stream:
            model_config = yaml.safe_load(stream)
        args.checkpoint = model_config["checkpoint"]
        args.revision = args.revision or model_config.get("revision")
        args.code_revision = args.code_revision or model_config.get("code_revision")
        args.suite = args.suite or model_config.get("suite")
    args.suite = args.suite or "libero_spatial"
    if args.suite not in MAX_STEPS:
        cli.error(f"Unknown suite: {args.suite}")
    if args.backend == "openvla" and not args.checkpoint:
        cli.error("--model-config or --checkpoint is required for OpenVLA")
    if min(args.task_id, args.initial_state, args.episode_num) < 0 or args.replan_steps < 1:
        cli.error("Task/state IDs must be nonnegative and replan steps positive")
    if args.max_steps is not None and args.max_steps < 1:
        cli.error("--max-steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    prepare_libero(args.libero_root, args.output)
    policy = OpenVLA(args) if args.backend == "openvla" else None
    result = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "policy": policy.metadata if policy is not None else None}

    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    actions, frames = [], []
    success = False
    completed = False
    trajectory = None
    target_positions = []
    prompt_record = {
        "task_suite_name": args.suite, "task_id": args.task_id,
        "episode_num": args.episode_num, "task_episode_idx": args.initial_state,
        "initial_state": args.initial_state, "task_description": task.language,
        "token_selection": "all_forward_positions",
    }
    if args.collection or steered or args.target_object:
        (args.output / "prompt_records.jsonl").write_text(json.dumps(prompt_record) + "\n")
        trajectory = (args.output / "trajectory_records.jsonl").open("w")
        def stop_on_sigterm(_signum, _frame):
            raise SystemExit(0)
        previous_sigterm_handler = signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(initial_states[args.initial_state])
        for _ in range(10):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        np.savez(args.output / "initial_observation.npz", **{
            key: obs[key] for key in ("agentview_image", "robot0_eye_in_hand_image",
                                     "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")})
        if args.backend == "openpi":
            policy = OpenPI(args)
            result["policy"] = policy.metadata
        budget = args.max_steps or MAX_STEPS[args.suite]
        for step in range(budget):
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            if trajectory is not None:
                policy.context = {**prompt_record, "step_in_episode": step, "env_t": step + 10}
            action = np.asarray(policy.action(obs, task.language))
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError(f"Invalid action: {action}")
            actions.append(action.copy())
            obs, _, done, _ = env.step(action.tolist())
            if trajectory is not None:
                behavior = {}
                if args.target_object:
                    from event_sae.behavior import object_poses_from_observation

                    behavior["object_poses"] = object_poses_from_observation(obs)
                    target_positions.append(behavior["object_poses"][args.target_object]["position"])
                trajectory.write(json.dumps({
                    **policy.context, "eef_pos": obs["robot0_eef_pos"].tolist(),
                    "eef_quat": obs["robot0_eef_quat"].tolist(),
                    "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
                    "action": action.tolist(), "gripper_action": float(action[-1]), "done": bool(done),
                    **behavior,
                }) + "\n")
            if done:
                success = True
                break
        completed = True
    finally:
        env.close()
        if trajectory is not None:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
            trajectory.close()
        if args.collection:
            if args.backend == "openvla" or hasattr(policy, "context"):
                # Retain normally completed episodes even when the task failed.
                policy.finish_collection(keep=completed)
            if args.backend == "openpi" and policy is not None:
                (args.output / "chunk_records.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in policy.chunk_records)
                )
        if args.backend == "openvla" and policy.steer_handle is not None:
            policy.steer_handle.remove()
    np.save(args.output / "actions.npy", np.asarray(actions))
    imageio.mimwrite(args.output / "rollout.mp4", frames, fps=20)
    result.update(task=task.language, steps=len(actions), success=success)
    if args.target_object:
        from event_sae.behavior import maximum_lift

        result["behavior"] = {"target_object": args.target_object,
                              "max_lift_height_m": maximum_lift(target_positions)}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
