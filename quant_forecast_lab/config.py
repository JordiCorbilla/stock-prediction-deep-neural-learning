"""Shared lightweight configuration contracts."""

from __future__ import annotations

DEFAULT_MODEL_VERSION = "v7"
DEFAULT_USE_RETURNS = False
DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.15

_DELTA_MODELS = {"v3", "v5", "v7", "v8"}
_TREND_MODELS = {"v6"}


def validate_model_options(model_version: str, use_returns: bool) -> None:
    """Raise ValueError for incompatible model and target combinations."""
    if model_version not in {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"}:
        raise ValueError(f"Unknown model version: {model_version}")
    if model_version == "v9" and not use_returns:\n        raise ValueError("v9 is a return-first model and requires --use_returns=true.")\n    if use_returns and model_version in (_DELTA_MODELS | _TREND_MODELS):
        raise ValueError(
            f"{model_version} predicts deltas/residuals and cannot be combined with --use_returns=true."
        )
