"""Time-series validation primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardFold:
    train: slice
    validation: slice | None
    test: slice


def expanding_window_splits(
    n_samples: int,
    min_train_size: int,
    test_size: int,
    *,
    validation_size: int = 0,
    step_size: int | None = None,
    gap: int = 0,
):
    """Yield chronological expanding-window train, validation and test slices."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if min_train_size <= 0 or test_size <= 0:
        raise ValueError("min_train_size and test_size must be positive.")
    if validation_size < 0 or gap < 0:
        raise ValueError("validation_size and gap cannot be negative.")

    step = test_size if step_size is None else step_size
    if step <= 0:
        raise ValueError("step_size must be positive.")

    train_end = min_train_size
    while True:
        validation_start = train_end
        validation_end = validation_start + validation_size
        test_start = validation_end + gap
        test_end = test_start + test_size
        if test_end > n_samples:
            break

        validation = slice(validation_start, validation_end) if validation_size else None
        yield WalkForwardFold(
            train=slice(0, train_end),
            validation=validation,
            test=slice(test_start, test_end),
        )
        train_end += step
