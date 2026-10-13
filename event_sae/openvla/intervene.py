"""SAE reconstruction and residual-preserving feature scaling for OpenVLA."""

from pathlib import Path

import torch

from event_sae.sae import load_batch_topk_sae


def apply_sae_hook(model, config):
    if config["mode"] == "native":
        return None
    device = str(model.language_model.lm_head.weight.device)
    sae, sae_config = load_batch_topk_sae(Path(config["sae_checkpoint"]), device=device)
    trainer = sae_config["trainer"]
    # Accept the original name in existing checkpoints.
    if trainer.get("submodule_name") not in {"resid_post", "post_mlp_residual"}:
        raise ValueError("Expected a resid_post SAE checkpoint")
    layer_idx = config["layer_idx"]
    decoder_layers = model.language_model.model.layers
    if not 0 <= layer_idx < len(decoder_layers):
        raise IndexError(f"Layer index out of range: {layer_idx}")
    if config["mode"] == "feature":
        feature_idx = config["feature_id"]
        if not 0 <= feature_idx < trainer["dict_size"]:
            raise IndexError(f"Feature index out of range: {feature_idx}")
    start_step = config.get("hook_start_step", 0)

    def hook_fn(module, inputs, output):
        del module, inputs
        context = model._sae_hook_context
        if context["step_in_episode"] < start_step:
            return output
        hidden = output[0]
        if hidden.ndim != 3 or hidden.shape[-1] != trainer["activation_dim"]:
            raise ValueError(f"Hidden shape {tuple(hidden.shape)} incompatible with SAE")
        flat = hidden.reshape(-1, hidden.shape[-1]).to(dtype=torch.float32)
        if config["mode"] == "reconstruction":
            updated = sae(flat)
        else:
            encoded = sae.encode(flat)
            perturbed = encoded.clone()
            perturbed[:, feature_idx] = perturbed[:, feature_idx] * float(config["feature_alpha"])
            recon_orig = sae.decode(encoded)
            recon_pert = sae.decode(perturbed)
            updated = flat + (recon_pert - recon_orig)
        updated = updated.to(dtype=hidden.dtype).reshape_as(hidden)
        return (updated, *output[1:])

    return decoder_layers[layer_idx].register_forward_hook(hook_fn)
