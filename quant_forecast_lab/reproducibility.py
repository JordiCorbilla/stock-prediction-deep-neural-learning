"""Reproducibility helpers shared by training entry points."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy and TensorFlow when TensorFlow is available.

    ``PYTHONHASHSEED`` must be set before the interpreter starts to affect the
    current process; assigning it here only propagates the seed to child
    processes launched afterwards.
    """
    seed = int(seed)
    # Hash randomization is fixed at interpreter startup for the current process.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:
        return seed

    tf.random.set_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass
    return seed
