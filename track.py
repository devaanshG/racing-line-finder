"""
Track module.

Responsibilities:
- Load track CSV (left_x, left_y, right_x, right_y per gate)
- Compute the raw centerline path
"""

import numpy as np
import pandas as pd

TRACK_COLUMNS = ["left_x", "left_y", "right_x", "right_y"]


def load_track(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and validate a track CSV.

    Expected columns: left_x, left_y, right_x, right_y
    Each row is one gate — left and right cone positions.

    Returns
    -------
    left_cones, right_cones : np.ndarray of shape (N, 2)
    """
    df = pd.read_csv(path)
    missing = [c for c in TRACK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Track CSV missing columns: {missing}. "
            f"Expected: {TRACK_COLUMNS}"
        )

    left = df[["left_x", "left_y"]].to_numpy(dtype=float)
    right = df[["right_x", "right_y"]].to_numpy(dtype=float)

    if len(left) < 3:
        raise ValueError(
            f"Need at least 3 gate pairs to define a track, got {len(left)}."
        )
    if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError("Track data contains NaN or infinite values.")

    return left, right


def gate_midpoints(left_cones: np.ndarray, right_cones: np.ndarray) -> np.ndarray:
    """
    Return the midpoint of each gate as a (N, 2) array.
    This is the naive centerline — no optimisation applied.
    """
    return (left_cones + right_cones) / 2.0
