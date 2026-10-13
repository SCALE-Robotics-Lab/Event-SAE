# Modified by the Event-SAE authors.
import dataclasses
import datetime as dt
import enum
import logging
import os
import signal
import sys
from pathlib import Path

import tyro
import yaml

from openpi.models import gemma as _gemma
from openpi.models import pi0_config as _pi0_config
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.sae_collection import collector as _sae_collector
from openpi.sae_collection import decoder_vector as _sae_decoder_vector
from openpi.sae_collection import reconstruction as _sae_reconstruction
from openpi.sae_collection import runtime as _sae_runtime
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi05_libero").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi05_libero/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class SAECollection:
    """Optional dense activation collection."""

    enabled: bool = False
    output_root: str = "outputs/openpi/activations"
    # Directory name under output_root; existing names get a numbered suffix unless appending.
    run_name: str | None = None
    # Reuse the same output_root/run_name and collection settings, with new episode IDs.
    append_to_existing: bool = False
    capture_target: str = "action_expert"
    layer_indices: str = "0,5,11,17"
    flush_every_rows: int = 50_000
    replan_steps: int | None = None


@dataclasses.dataclass
class SAEReconstruction:
    """Optional SAE reconstruction replacement hook."""

    enabled: bool = False
    ae_path: str = ""
    capture_target: str = "action_expert"
    layer_idx: int = 17
    feature_indices: str = ""
    feature_alpha: float = 0.0
    active_feature_drop_count: int = 0
    active_feature_drop_seed: int = 0


@dataclasses.dataclass
class SAEDecoderVectorAdd:
    """Add SAE decoder directions or a norm-matched random direction."""

    enabled: bool = False
    ae_path: str = ""
    capture_target: str = "action_expert"
    layer_idx: int = 0
    feature_indices: str = ""
    alpha: float = 150.0
    random_baseline: bool = False
    random_baseline_seed: int = 0


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.LIBERO

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # YAML with the OpenPI training config name and checkpoint path.
    model_config: Path | None = None
    collect_config: Path | None = None
    steer_config: Path | None = None

    # Optional dense activation collection for SAE training/analysis.
    sae: SAECollection = dataclasses.field(default_factory=SAECollection)

    # Optional SAE reconstruction hook for closed-loop faithfulness eval.
    sae_recon: SAEReconstruction = dataclasses.field(default_factory=SAEReconstruction)

    # Optional SAE decoder-vector-add (activation-steering) hook.
    sae_dvec: SAEDecoderVectorAdd = dataclasses.field(default_factory=SAEDecoderVectorAdd)

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _resolve_policy_spec(args: Args) -> tuple[_config.TrainConfig, str]:
    if args.model_config is not None:
        if isinstance(args.policy, Checkpoint):
            raise ValueError("Use either --model-config or policy:checkpoint.")
        with args.model_config.open() as stream:
            model_config = yaml.safe_load(stream)
        return _config.get_config(model_config["config"]), model_config["checkpoint"]
    match args.policy:
        case Checkpoint():
            return _config.get_config(args.policy.config), args.policy.dir
        case Default():
            if checkpoint := DEFAULT_CHECKPOINT.get(args.env):
                return _config.get_config(checkpoint.config), checkpoint.dir
    raise ValueError(f"Unsupported environment mode: {args.env}")


def _target_variant_and_spec(
    train_config: _config.TrainConfig,
    capture_target: str,
) -> tuple[str, _sae_collector.CaptureTargetSpec]:
    if capture_target == "action_expert":
        target_variant = train_config.model.action_expert_variant
    elif capture_target == "paligemma":
        target_variant = train_config.model.paligemma_variant
    else:
        raise ValueError(f"Unsupported capture_target={capture_target!r}. Expected 'action_expert' or 'paligemma'.")
    return target_variant, _sae_collector.CAPTURE_TARGETS[capture_target]


def _parse_int_list(raw: str) -> tuple[int, ...]:
    if not raw:
        return ()
    return tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())


def _make_sae_collection(
    train_config: _config.TrainConfig,
    checkpoint_dir: str,
    sae_args: SAECollection,
) -> tuple[_sae_collector.ActivationCollector | None, dict | None]:
    if not sae_args.enabled:
        return None, None
    if not isinstance(train_config.model, _pi0_config.Pi0Config):
        raise ValueError("Dense SAE collection currently supports Pi0/Pi0.5 JAX checkpoints only.")

    candidate_weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(candidate_weight_path):
        raise ValueError("Dense SAE collection is not implemented for PyTorch checkpoints in this workspace.")

    target_variant, _ = _target_variant_and_spec(train_config, sae_args.capture_target)

    target_config = _gemma.get_config(target_variant)
    state_token_offset = 0 if train_config.model.pi05 else 1
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = sae_args.run_name or f"{train_config.name}_{sae_args.capture_target}_{timestamp}"
    collector = _sae_collector.ActivationCollector(
        _sae_collector.SAECollectionConfig(
            output_root=sae_args.output_root,
            run_name=run_name,
            append_to_existing=sae_args.append_to_existing,
            capture_target=sae_args.capture_target,
            layer_indices=_sae_collector.parse_layer_indices(sae_args.layer_indices),
            flush_every_rows=int(sae_args.flush_every_rows),
            model_type=train_config.model.model_type.value,
            pi05=bool(train_config.model.pi05),
            action_horizon=int(train_config.model.action_horizon),
            action_dim=int(train_config.model.action_dim),
            state_token_offset=state_token_offset,
            action_token_offset=state_token_offset,
            replan_steps=None if sae_args.replan_steps is None else int(sae_args.replan_steps),
        ),
        policy_config_name=train_config.name,
        checkpoint_dir=checkpoint_dir,
        model_depth=target_config.depth,
        d_model=target_config.width,
    )
    os.environ["OPENPI_ENABLE_SAE_COLLECTION"] = "1"
    _sae_runtime.set_active_collector(collector)
    sample_kwargs = {
        "sae_capture_layer_mask": collector.layer_mask,
        "sae_capture_expert_idx": collector.expert_idx,
    }
    return collector, {
        "sample_kwargs": sample_kwargs,
        "metadata": collector.server_metadata(),
    }


def _make_sae_reconstruction(
    train_config: _config.TrainConfig,
    checkpoint_dir: str,
    recon_args: SAEReconstruction,
) -> dict | None:
    if not recon_args.enabled:
        return None
    if not isinstance(train_config.model, _pi0_config.Pi0Config):
        raise ValueError("SAE reconstruction currently supports Pi0/Pi0.5 JAX checkpoints only.")

    candidate_weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(candidate_weight_path):
        raise ValueError("SAE reconstruction is not implemented for PyTorch checkpoints in this workspace.")
    if not recon_args.ae_path:
        raise ValueError("--sae-recon.ae-path is required when SAE reconstruction is enabled.")

    target_variant, target_spec = _target_variant_and_spec(train_config, recon_args.capture_target)
    target_config = _gemma.get_config(target_variant)
    if recon_args.layer_idx < 0 or recon_args.layer_idx >= target_config.depth:
        raise ValueError(
            f"SAE reconstruction layer_idx={recon_args.layer_idx} is outside model depth {target_config.depth}."
        )

    import torch

    state_dict = torch.load(recon_args.ae_path, map_location="cpu")
    required_keys = ("encoder.weight", "encoder.bias", "decoder.weight", "b_dec", "k", "threshold")
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        raise ValueError(f"SAE checkpoint {recon_args.ae_path} is missing keys: {missing_keys}")

    state = _sae_reconstruction.make_state_from_arrays(
        ae_path=recon_args.ae_path,
        capture_target=recon_args.capture_target,
        expert_idx=target_spec.expert_idx,
        layer_idx=int(recon_args.layer_idx),
        encoder_weight=state_dict["encoder.weight"].detach().cpu().numpy(),
        encoder_bias=state_dict["encoder.bias"].detach().cpu().numpy(),
        decoder_weight=state_dict["decoder.weight"].detach().cpu().numpy(),
        b_dec=state_dict["b_dec"].detach().cpu().numpy(),
        k=int(state_dict["k"].item()),
        threshold=float(state_dict["threshold"].item()),
        feature_indices=_parse_int_list(recon_args.feature_indices),
        feature_alpha=float(recon_args.feature_alpha),
        active_feature_drop_count=int(recon_args.active_feature_drop_count),
        active_feature_drop_seed=int(recon_args.active_feature_drop_seed),
    )
    if state.activation_dim != target_config.width:
        raise ValueError(
            f"SAE activation_dim={state.activation_dim} does not match {target_variant} width={target_config.width}."
        )

    _sae_reconstruction.set_active_reconstruction(state)
    if _sae_reconstruction.requires_hidden_state_edit():
        os.environ["OPENPI_ENABLE_SAE_RECONSTRUCTION"] = "1"
    else:
        os.environ.pop("OPENPI_ENABLE_SAE_RECONSTRUCTION", None)
    return {"metadata": state.metadata()}


def _make_sae_decoder_vector(
    train_config: _config.TrainConfig,
    checkpoint_dir: str,
    dvec_args: SAEDecoderVectorAdd,
) -> dict | None:
    if not dvec_args.enabled:
        return None
    if not isinstance(train_config.model, _pi0_config.Pi0Config):
        raise ValueError("SAE decoder_vector_add currently supports Pi0/Pi0.5 JAX checkpoints only.")

    candidate_weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(candidate_weight_path):
        raise ValueError("SAE decoder_vector_add is not implemented for PyTorch checkpoints in this workspace.")
    if not dvec_args.ae_path:
        raise ValueError("--sae-dvec.ae-path is required when SAE decoder_vector_add is enabled.")

    feature_indices = _parse_int_list(dvec_args.feature_indices)
    if not feature_indices:
        raise ValueError("--sae-dvec.feature-indices must list at least one feature id.")

    target_variant, target_spec = _target_variant_and_spec(train_config, dvec_args.capture_target)
    target_config = _gemma.get_config(target_variant)
    if dvec_args.layer_idx < 0 or dvec_args.layer_idx >= target_config.depth:
        raise ValueError(
            f"SAE decoder_vector_add layer_idx={dvec_args.layer_idx} is outside model depth {target_config.depth}."
        )

    import torch

    state_dict = torch.load(dvec_args.ae_path, map_location="cpu")
    if "decoder.weight" not in state_dict:
        raise ValueError(f"SAE checkpoint {dvec_args.ae_path} is missing decoder.weight.")

    state = _sae_decoder_vector.make_state_from_arrays(
        ae_path=dvec_args.ae_path,
        capture_target=dvec_args.capture_target,
        expert_idx=target_spec.expert_idx,
        layer_idx=int(dvec_args.layer_idx),
        decoder_weight=state_dict["decoder.weight"].detach().cpu().numpy(),
        feature_indices=feature_indices,
        alpha=float(dvec_args.alpha),
        random_baseline=bool(dvec_args.random_baseline),
        random_baseline_seed=int(dvec_args.random_baseline_seed),
    )
    if state.activation_dim != target_config.width:
        raise ValueError(
            f"SAE activation_dim={state.activation_dim} does not match {target_variant} width={target_config.width}."
        )

    _sae_decoder_vector.set_active_decoder_vector_add(state)
    if _sae_decoder_vector.requires_hidden_state_edit():
        os.environ["OPENPI_ENABLE_SAE_DECODER_VECTOR"] = "1"
    else:
        os.environ.pop("OPENPI_ENABLE_SAE_DECODER_VECTOR", None)
    return {"metadata": state.metadata()}


def _clear_sae_runtime(sae_collector: _sae_collector.ActivationCollector | None) -> None:
    if sae_collector is not None:
        sae_collector.close()
    _sae_runtime.clear_active_collector()
    _sae_reconstruction.clear_active_reconstruction()
    _sae_decoder_vector.clear_active_decoder_vector_add()
    os.environ.pop("OPENPI_ENABLE_SAE_COLLECTION", None)
    os.environ.pop("OPENPI_ENABLE_SAE_RECONSTRUCTION", None)
    os.environ.pop("OPENPI_ENABLE_SAE_DECODER_VECTOR", None)


def create_policy(args: Args) -> tuple[_policy.Policy, dict, _sae_collector.ActivationCollector | None]:
    """Create a policy from the given arguments."""
    steering = None
    if args.steer_config is not None:
        from event_sae.steering import load_steering_config

        steering = load_steering_config(args.steer_config)
        if steering["mode"] == "native":
            args.sae_recon = SAEReconstruction()
        else:
            args.sae_recon = SAEReconstruction(
                enabled=True, ae_path=steering["sae_checkpoint"],
                capture_target=steering["capture_target"], layer_idx=steering["layer_idx"],
                feature_indices=str(steering["feature_id"]) if steering["mode"] == "feature" else "",
                feature_alpha=steering.get("feature_alpha", 0.0),
            )
    train_config, checkpoint_dir = _resolve_policy_spec(args)
    sae_collector = None
    try:
        sae_collector, sae_payload = _make_sae_collection(train_config, checkpoint_dir, args.sae)
        sae_recon_payload = _make_sae_reconstruction(train_config, checkpoint_dir, args.sae_recon)
        sae_dvec_payload = _make_sae_decoder_vector(train_config, checkpoint_dir, args.sae_dvec)
        if sae_recon_payload is not None and sae_dvec_payload is not None:
            raise ValueError(
                "SAE reconstruction and SAE decoder_vector_add cannot both be enabled in the same server."
            )
        policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=args.default_prompt,
            sample_kwargs=None if sae_payload is None else sae_payload["sample_kwargs"],
        )
    except Exception:
        _clear_sae_runtime(sae_collector)
        raise
    metadata = dict(policy.metadata)
    if steering is not None:
        metadata["steering"] = steering
    if sae_payload is not None:
        metadata["sae_collection"] = sae_payload["metadata"]
        logging.info("Enabled SAE collection: %s", sae_payload["metadata"])
    if sae_recon_payload is not None:
        metadata["sae_reconstruction"] = sae_recon_payload["metadata"]
        logging.info("Enabled SAE reconstruction: %s", sae_recon_payload["metadata"])
    if sae_dvec_payload is not None:
        metadata["sae_decoder_vector_add"] = sae_dvec_payload["metadata"]
        logging.info("Enabled SAE decoder_vector_add: %s", sae_dvec_payload["metadata"])
    return policy, metadata, sae_collector


def main(args: Args, *, cli_args: list[str] | None = None) -> None:
    if args.collect_config is not None:
        with args.collect_config.open() as stream:
            collection_config = yaml.safe_load(stream)
        explicit_fields = {
            arg.split("=", 1)[0].split(".", 1)[1].replace("_", "-").removeprefix("no-").replace("-", "_")
            for arg in (cli_args or [])
            if arg.startswith("--sae.")
        }
        duplicates = explicit_fields.intersection(collection_config)
        if duplicates:
            fields = ", ".join(sorted(duplicates))
            raise SystemExit(
                f"Collection fields set in both --collect-config and --sae.*: {fields}. "
                "Specify each field in only one place."
            )
        args.sae = dataclasses.replace(args.sae, enabled=True, **collection_config)
    policy, policy_metadata, sae_collector = create_policy(args)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    def stop_on_sigterm(_signum, _frame):
        # Slurm/batch wrappers stop the server with SIGTERM after the client
        # exits. Unwind through finally so buffered SAE rows reach disk.
        raise SystemExit(0)

    previous_sigterm_handler = signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        _clear_sae_runtime(sae_collector)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    if args.steer_config is not None and any(
        arg.replace("_", "-").startswith("--sae-recon.") for arg in sys.argv[1:]
    ):
        raise SystemExit("Use either --steer-config or --sae-recon.* options, not both.")
    main(args, cli_args=sys.argv[1:])
