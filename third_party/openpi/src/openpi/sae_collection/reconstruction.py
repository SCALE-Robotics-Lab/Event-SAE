"""Pure JAX SAE reconstruction hook for closed-loop faithfulness eval."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class SAEReconstructionState:
    ae_path: str
    capture_target: str
    expert_idx: int
    layer_idx: int
    activation_dim: int
    dict_size: int
    k: int
    threshold: float
    feature_indices: tuple[int, ...]
    feature_alpha: float
    active_feature_drop_count: int
    active_feature_drop_seed: int
    encoder_weight: jax.Array
    encoder_bias: jax.Array
    decoder_weight: jax.Array
    b_dec: jax.Array

    def metadata(self) -> dict[str, Any]:
        if self.active_feature_drop_count > 0:
            mode = "active_feature_dropout_delta"
        elif self.feature_indices:
            mode = "latent_feature_scale_delta"
        else:
            mode = "thresholded_sae_reconstruction"
        return {
            "enabled": True,
            "ae_path": self.ae_path,
            "capture_target": self.capture_target,
            "expert_idx": self.expert_idx,
            "layer_idx": self.layer_idx,
            "activation_dim": self.activation_dim,
            "dict_size": self.dict_size,
            "k": self.k,
            "threshold": self.threshold,
            "feature_indices": list(self.feature_indices),
            "feature_alpha": self.feature_alpha,
            "active_feature_drop_count": self.active_feature_drop_count,
            "active_feature_drop_seed": self.active_feature_drop_seed,
            "hook_site": "resid_post",
            "mode": mode,
        }


_ACTIVE_RECONSTRUCTION: SAEReconstructionState | None = None


def set_active_reconstruction(state: SAEReconstructionState | None) -> None:
    global _ACTIVE_RECONSTRUCTION
    _ACTIVE_RECONSTRUCTION = state


def clear_active_reconstruction() -> None:
    set_active_reconstruction(None)


def is_enabled() -> bool:
    return _ACTIVE_RECONSTRUCTION is not None


def requires_hidden_state_edit() -> bool:
    """Return whether the active configuration must change the model graph."""
    state = _ACTIVE_RECONSTRUCTION
    if state is None:
        return False
    return not (
        state.feature_alpha == 1.0
        and (state.feature_indices or state.active_feature_drop_count > 0)
    )


def active_metadata() -> dict[str, Any] | None:
    if _ACTIVE_RECONSTRUCTION is None:
        return None
    return _ACTIVE_RECONSTRUCTION.metadata()


def make_state_from_arrays(
    *,
    ae_path: str,
    capture_target: str,
    expert_idx: int,
    layer_idx: int,
    encoder_weight: np.ndarray,
    encoder_bias: np.ndarray,
    decoder_weight: np.ndarray,
    b_dec: np.ndarray,
    k: int,
    threshold: float,
    feature_indices: tuple[int, ...] = (),
    feature_alpha: float = 0.0,
    active_feature_drop_count: int = 0,
    active_feature_drop_seed: int = 0,
) -> SAEReconstructionState:
    dict_size, activation_dim = encoder_weight.shape
    if decoder_weight.shape != (activation_dim, dict_size):
        raise ValueError(
            "Expected decoder_weight shape "
            f"({activation_dim}, {dict_size}), got {decoder_weight.shape}."
        )
    if encoder_bias.shape != (dict_size,):
        raise ValueError(f"Expected encoder_bias shape ({dict_size},), got {encoder_bias.shape}.")
    if b_dec.shape != (activation_dim,):
        raise ValueError(f"Expected b_dec shape ({activation_dim},), got {b_dec.shape}.")
    if any(feature_idx < 0 or feature_idx >= dict_size for feature_idx in feature_indices):
        raise ValueError(f"feature_indices out of range for dict_size={dict_size}: {feature_indices}")
    if active_feature_drop_count < 0 or active_feature_drop_count > dict_size:
        raise ValueError(
            f"Expected active_feature_drop_count in [0, {dict_size}], got {active_feature_drop_count}."
        )
    if feature_indices and active_feature_drop_count > 0:
        raise ValueError("feature_indices and active_feature_drop_count cannot both be set.")

    return SAEReconstructionState(
        ae_path=ae_path,
        capture_target=capture_target,
        expert_idx=int(expert_idx),
        layer_idx=int(layer_idx),
        activation_dim=int(activation_dim),
        dict_size=int(dict_size),
        k=int(k),
        threshold=float(threshold),
        feature_indices=tuple(int(feature_idx) for feature_idx in feature_indices),
        feature_alpha=float(feature_alpha),
        active_feature_drop_count=int(active_feature_drop_count),
        active_feature_drop_seed=int(active_feature_drop_seed),
        encoder_weight=jnp.asarray(encoder_weight, dtype=jnp.float32),
        encoder_bias=jnp.asarray(encoder_bias, dtype=jnp.float32),
        decoder_weight=jnp.asarray(decoder_weight, dtype=jnp.float32),
        b_dec=jnp.asarray(b_dec, dtype=jnp.float32),
    )


def maybe_reconstruct_hidden_states(
    hidden_states: jax.Array,
    *,
    layer_idx: jax.Array,
    expert_idx: int,
) -> jax.Array:
    state = _ACTIVE_RECONSTRUCTION
    if state is None:
        return hidden_states
    if int(expert_idx) != int(state.expert_idx):
        return hidden_states

    hidden_dim = int(hidden_states.shape[-1])
    if hidden_dim != int(state.activation_dim):
        raise ValueError(
            f"SAE reconstruction target {state.capture_target!r} expected activation_dim={state.activation_dim}, "
            f"but got hidden state last dim={hidden_dim} for expert_idx={expert_idx}."
        )

    if not requires_hidden_state_edit():
        return hidden_states

    should_reconstruct = layer_idx == jnp.int32(state.layer_idx)

    def _replace(value: jax.Array) -> jax.Array:
        original_dtype = value.dtype
        flat = jnp.reshape(value.astype(jnp.float32), (-1, state.activation_dim))
        pre_acts = jax.nn.relu(
            jnp.matmul(
                flat - state.b_dec,
                state.encoder_weight.T,
                precision=jax.lax.Precision.HIGHEST,
            )
            + state.encoder_bias
        )
        features = jnp.where(pre_acts > jnp.asarray(state.threshold, dtype=jnp.float32), pre_acts, 0.0)
        reconstruction = jnp.matmul(
            features,
            state.decoder_weight.T,
            precision=jax.lax.Precision.HIGHEST,
        ) + state.b_dec
        if state.feature_indices:
            feature_indices = jnp.asarray(state.feature_indices, dtype=jnp.int32)
            original_feature_values = features[:, feature_indices]
            perturbed_features = features.at[:, feature_indices].set(
                original_feature_values * jnp.asarray(state.feature_alpha, dtype=jnp.float32)
            )
            perturbed_reconstruction = jnp.matmul(
                perturbed_features,
                state.decoder_weight.T,
                precision=jax.lax.Precision.HIGHEST,
            ) + state.b_dec
            reconstruction = flat + (perturbed_reconstruction - reconstruction)
        elif state.active_feature_drop_count > 0:
            active = features > 0.0
            row_ids = jnp.arange(features.shape[0], dtype=jnp.float32)[:, None]
            feature_ids = jnp.arange(state.dict_size, dtype=jnp.float32)[None, :]
            seed = jnp.asarray(state.active_feature_drop_seed, dtype=jnp.float32)
            # Deterministic pseudo-random scores avoid threading a PRNG through the policy call.
            randomish = jnp.sin((feature_ids + 1.0) * 12.9898 + (row_ids + 1.0) * 78.233 + seed * 0.0174533)
            randomish = randomish * 43758.5453
            randomish = randomish - jnp.floor(randomish)
            masked_scores = jnp.where(active, randomish, jnp.inf)
            drop_indices = jnp.argsort(masked_scores, axis=1)[:, : state.active_feature_drop_count]
            drop_rows = jnp.arange(features.shape[0], dtype=jnp.int32)[:, None]
            drop_mask = jnp.zeros_like(features, dtype=bool).at[drop_rows, drop_indices].set(True)
            drop_mask = jnp.logical_and(drop_mask, active)
            perturbed_features = jnp.where(
                drop_mask,
                features * jnp.asarray(state.feature_alpha, dtype=jnp.float32),
                features,
            )
            perturbed_reconstruction = jnp.matmul(
                perturbed_features,
                state.decoder_weight.T,
                precision=jax.lax.Precision.HIGHEST,
            ) + state.b_dec
            reconstruction = flat + (perturbed_reconstruction - reconstruction)
        return jnp.reshape(reconstruction, value.shape).astype(original_dtype)

    return jax.lax.cond(
        should_reconstruct,
        _replace,
        lambda value: value,
        hidden_states,
    )
