"""Dense SAE activation collection helpers for OpenPI rollout runs."""

__all__ = [
    "ActivationCollector",
    "SAECollectionConfig",
    "clear_active_collector",
    "emit_activation_callback",
    "finish_request",
    "lookup_activation_rows",
    "request_context",
    "set_active_collector",
    "validate_run_dir",
]

_COLLECTOR_EXPORTS = {
    "ActivationCollector",
    "SAECollectionConfig",
    "lookup_activation_rows",
    "validate_run_dir",
}
_RUNTIME_EXPORTS = {
    "clear_active_collector",
    "emit_activation_callback",
    "finish_request",
    "request_context",
    "set_active_collector",
}


def __getattr__(name: str):
    if name in _COLLECTOR_EXPORTS:
        from openpi.sae_collection import collector

        return getattr(collector, name)
    if name in _RUNTIME_EXPORTS:
        from openpi.sae_collection import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
