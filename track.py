"""
Track module.

Responsibilities:
- Load track CSV (left_x, left_y, right_x, right_y per gate)
- Compute gate midpoints (naive centerline)
- Compute gate widths and directions
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

    left  = df[["left_x",  "left_y"]].to_numpy(dtype=float)
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


def gate_widths(left_cones: np.ndarray, right_cones: np.ndarray) -> np.ndarray:
    """Return the width of each gate (distance between left and right cone)."""
    return np.linalg.norm(right_cones - left_cones, axis=1)


def is_loop_closed(left_cones: np.ndarray, right_cones: np.ndarray,
                   tol: float = None) -> bool:
    """
    Return True if the track forms a closed loop.

    The track is considered closed when the first and last gate midpoints
    are within `tol` of each other.  Default tolerance = 5% of the mean
    gate-to-gate spacing along the centerline.
    """
    centre = gate_midpoints(left_cones, right_cones)
    if tol is None:
        spacings = np.linalg.norm(np.diff(centre, axis=0), axis=1)
        tol = 5.0 * float(np.mean(spacings))
    gap = np.linalg.norm(centre[-1] - centre[0])
    return bool(gap < tol)
