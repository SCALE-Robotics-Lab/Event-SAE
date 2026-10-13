"""Runtime helpers for SAE activation collection during policy inference."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
from typing import TYPE_CHECKING

import jax
import numpy as np

if TYPE_CHECKING:
    from openpi.sae_collection.collector import ActivationCollector


_ACTIVE_COLLECTOR: ActivationCollector | None = None


def set_active_collector(collector: ActivationCollector | None) -> None:
    global _ACTIVE_COLLECTOR
    _ACTIVE_COLLECTOR = collector


def clear_active_collector() -> None:
    set_active_collector(None)


@contextlib.contextmanager
def request_context(context: dict | None) -> Iterator[ActivationCollector | None]:
    collector = _ACTIVE_COLLECTOR
    if collector is None or context is None:
        yield None
        return
    if "episode_num" not in context:
        raise ValueError("SAE collection context requires episode_num.")

    collector.begin_request(context)
    try:
        yield collector
    finally:
        collector.clear_request()


def finish_request(outputs: dict) -> None:
    collector = _ACTIVE_COLLECTOR
    if collector is not None:
        collector.finish_request(outputs)


def handle_control(command: dict) -> dict:
    collector = _ACTIVE_COLLECTOR
    if collector is None:
        return {
            "status": "error",
            "message": "No active SAE collector is registered on the policy server.",
        }
    return collector.handle_control(command)


def emit_activation_callback(
    layer_idx,
    expert_idx,
    forward_type,
    forward_step_idx,
    hidden_states,
) -> np.int32:
    collector = _ACTIVE_COLLECTOR
    if collector is not None:
        collector.record_activation(
            layer_idx=int(np.asarray(layer_idx)),
            expert_idx=int(np.asarray(expert_idx)),
            forward_type=int(np.asarray(forward_type)),
            forward_step_idx=int(np.asarray(forward_step_idx)),
            hidden_states=np.asarray(hidden_states),
        )
    return np.int32(0)


def io_capture_activation(
    hidden_states,
    *,
    layer_idx,
    expert_idx,
    forward_type,
    forward_step_idx,
):
    """Emit a host callback for a selected activation tensor."""

    shape = jax.ShapeDtypeStruct((), np.int32)
    return jax.experimental.io_callback(
        emit_activation_callback,
        shape,
        layer_idx,
        expert_idx,
        forward_type,
        forward_step_idx,
        hidden_states,
        ordered=True,
    )
