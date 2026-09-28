"""Experiment provenance helpers."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_sha(cwd=None) -> str | None:
    """Return the current Git commit if the working tree is a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def runtime_metadata(cwd=None) -> dict[str, str | None]:
    """Capture versions needed to interpret a saved experiment."""
    try:
        import tensorflow as tf

        tensorflow_version = tf.__version__
    except ImportError:
        tensorflow_version = None

    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "tensorflow_version": tensorflow_version,
        "git_commit": git_commit_sha(cwd=cwd),
    }
