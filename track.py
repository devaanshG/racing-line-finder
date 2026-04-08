"""
Track module.

Responsibilities:
- Load left/right cone CSVs
- Generate gates (paired left/right cones by row index)
- Compute the raw centerline path
"""

import numpy as np
import pandas as pd


def load_cones(left_path: str, right_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and validate cone CSVs.

    Each CSV must have columns 'x' and 'y'. Row i in the left CSV and row i in
    the right CSV form gate i — no matching is performed.

    Returns
    -------
    left_cones, right_cones : np.ndarray of shape (N, 2)
    """
    left = pd.read_csv(left_path)[["x", "y"]].to_numpy(dtype=float)
    right = pd.read_csv(right_path)[["x", "y"]].to_numpy(dtype=float)

    if len(left) != len(right):
        raise ValueError(
            f"Cone count mismatch: left has {len(left)} rows, right has {len(right)}."
        )
    if len(left) < 3:
        raise ValueError(
            f"Need at least 3 gate pairs to define a track, got {len(left)}."
        )
    if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError("Cone data contains NaN or infinite values.")

    return left, right


def gate_midpoints(left_cones: np.ndarray, right_cones: np.ndarray) -> np.ndarray:
    """
    Return the midpoint of each gate as a (N, 2) array.
    This is the naive centerline — no optimisation applied.
    """
    return (left_cones + right_cones) / 2.0
