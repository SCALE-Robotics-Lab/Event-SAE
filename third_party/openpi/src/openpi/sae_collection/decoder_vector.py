"""Pure JAX SAE decoder-vector-add (activation-steering) hook.

Decoder-vector addition:

    h' = h + alpha * sum_{f in F} W_dec[:, f]

The scaled steering vector is precomputed on the host so the JAX side is a
single broadcast add at the post-MLP residual of the matched (expert, layer).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class SAEDecoderVectorAddState:
    ae_path: str
    capture_target: str
    expert_idx: int
    layer_idx: int
    activation_dim: int
    dict_size: int
    feature_indices: tuple[int, ...]
    alpha: float
    scaled_steering_vector: jax.Array  # shape (activation_dim,), already-scaled steering vector
    steering_vector_l2_norm: float
    mode: str  # "decoder_vector_add" or "random_unit_norm_matched"
    random_baseline: bool
    random_baseline_seed: int

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": True,
            "ae_path": self.ae_path,
            "capture_target": self.capture_target,
            "expert_idx": self.expert_idx,
            "layer_idx": self.layer_idx,
            "activation_dim": self.activation_dim,
            "dict_size": self.dict_size,
            "feature_indices": list(self.feature_indices),
            "alpha": self.alpha,
            "steering_vector_l2_norm": self.steering_vector_l2_norm,
            "hook_site": "resid_post",
            "mode": self.mode,
        }
        if self.random_baseline:
            payload["random_baseline"] = True
            payload["random_baseline_seed"] = self.random_baseline_seed
        return payload


_ACTIVE_DECODER_VECTOR_ADD: SAEDecoderVectorAddState | None = None


def set_active_decoder_vector_add(state: SAEDecoderVectorAddState | None) -> None:
    global _ACTIVE_DECODER_VECTOR_ADD
    _ACTIVE_DECODER_VECTOR_ADD = state


def clear_active_decoder_vector_add() -> None:
    set_active_decoder_vector_add(None)


def is_enabled() -> bool:
    return _ACTIVE_DECODER_VECTOR_ADD is not None


def requires_hidden_state_edit() -> bool:
    return _ACTIVE_DECODER_VECTOR_ADD is not None and _ACTIVE_DECODER_VECTOR_ADD.steering_vector_l2_norm != 0.0


def active_metadata() -> dict[str, Any] | None:
    if _ACTIVE_DECODER_VECTOR_ADD is None:
        return None
    return _ACTIVE_DECODER_VECTOR_ADD.metadata()


def make_state_from_arrays(
    *,
    ae_path: str,
    capture_target: str,
    expert_idx: int,
    layer_idx: int,
    decoder_weight: np.ndarray,
    feature_indices: tuple[int, ...],
    alpha: float,
    random_baseline: bool = False,
    random_baseline_seed: int = 0,
) -> SAEDecoderVectorAddState:
    if not feature_indices:
        raise ValueError("decoder_vector_add requires at least one feature index.")
    activation_dim, dict_size = decoder_weight.shape
    if any(feature_idx < 0 or feature_idx >= dict_size for feature_idx in feature_indices):
        raise ValueError(f"feature_indices out of range for dict_size={dict_size}: {feature_indices}")

    decoder_columns = decoder_weight[:, list(feature_indices)].astype(np.float32)
    steering_vector = decoder_columns.sum(axis=1)
    decoder_scaled_norm = float(np.linalg.norm(float(alpha) * steering_vector))

    if random_baseline:
        rng = np.random.default_rng(int(random_baseline_seed))
        raw = rng.standard_normal(int(activation_dim)).astype(np.float32)
        raw_norm = float(np.linalg.norm(raw))
        if raw_norm == 0.0:
            raise ValueError("Generated a zero-norm random vector; pick a different seed.")
        unit = raw / raw_norm
        scaled_steering_vector = decoder_scaled_norm * unit
        mode = "random_unit_norm_matched"
    else:
        scaled_steering_vector = float(alpha) * steering_vector
        mode = "decoder_vector_add"

    return SAEDecoderVectorAddState(
        ae_path=ae_path,
        capture_target=capture_target,
        expert_idx=int(expert_idx),
        layer_idx=int(layer_idx),
        activation_dim=int(activation_dim),
        dict_size=int(dict_size),
        feature_indices=tuple(int(feature_idx) for feature_idx in feature_indices),
        alpha=float(alpha),
        scaled_steering_vector=jnp.asarray(scaled_steering_vector, dtype=jnp.float32),
        steering_vector_l2_norm=float(np.linalg.norm(scaled_steering_vector)),
        mode=mode,
        random_baseline=bool(random_baseline),
        random_baseline_seed=int(random_baseline_seed),
    )


def maybe_add_decoder_vector(hidden_states: jax.Array, *, layer_idx: jax.Array, expert_idx: int) -> jax.Array:
    state = _ACTIVE_DECODER_VECTOR_ADD
    if state is None:
        return hidden_states
    if int(expert_idx) != int(state.expert_idx):
        return hidden_states
    if state.steering_vector_l2_norm == 0.0:
        return hidden_states

    hidden_dim = int(hidden_states.shape[-1])
    if hidden_dim != int(state.activation_dim):
        raise ValueError(
            f"SAE decoder_vector_add target {state.capture_target!r} expected activation_dim={state.activation_dim}, "
            f"but got hidden state last dim={hidden_dim} for expert_idx={expert_idx}."
        )

    should_apply = layer_idx == jnp.int32(state.layer_idx)

    def _add(value: jax.Array) -> jax.Array:
        original_dtype = value.dtype
        flat = jnp.reshape(value.astype(jnp.float32), (-1, state.activation_dim))
        updated = flat + state.scaled_steering_vector[None, :]
        return jnp.reshape(updated, value.shape).astype(original_dtype)

    return jax.lax.cond(
        should_apply,
        _add,
        lambda value: value,
        hidden_states,
    )
