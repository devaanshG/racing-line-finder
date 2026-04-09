"""
Generates sample left_cones.csv and right_cones.csv for a simple FSAE-style circuit.

Track layout (counter-clockwise):
  - Long straight along the bottom
  - Right-hand hairpin at the east end
  - Return straight along the top
  - Left-hand hairpin at the west end
  - Short chicane on the bottom straight

Track half-width: 1.5 m
Run with: python data/generate_sample.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

HALF_WIDTH = 1.5  # metres either side of the centreline


def arc(cx: float, cy: float, r: float, t_start: float, t_end: float, n: int):
    """Return (x, y) points along a circular arc, counter-clockwise."""
    ts = np.linspace(t_start, t_end, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(ts), cy + r * np.sin(ts)])


def offset_path(path: np.ndarray, width: float):
    """
    Offset a closed path inward (left) and outward (right) by `width`.
    Returns (left_cones, right_cones) — left is the inner boundary.
    """
    n = len(path)
    lefts, rights = [], []

    for i in range(n):
        prev = path[(i - 1) % n]
        curr = path[i]
        nxt = path[(i + 1) % n]

        # Tangent as average of incoming and outgoing directions
        t1 = curr - prev
        t2 = nxt - curr
        tangent = t1 + t2
        norm = np.linalg.norm(tangent)
        if norm < 1e-9:
            tangent = t2
            norm = np.linalg.norm(tangent)
        tangent /= norm

        # Left-pointing normal (90° CCW from tangent)
        normal = np.array([-tangent[1], tangent[0]])

        lefts.append(curr + width * normal)
        rights.append(curr - width * normal)

    return np.array(lefts), np.array(rights)


def build_centreline(n_gates: int) -> np.ndarray:
    """
    Assemble a closed centreline resampled to exactly n_gates points.
    Layout: two straights + two hairpins.
    """
    from scipy.interpolate import splprep, splev

    # Build a dense base shape, then resample to the requested count
    points = []

    # Bottom straight: west (-18, -8) → east (18, -8)
    xs = np.linspace(-18, 18, 60, endpoint=False)
    points.append(np.column_stack([xs, np.full_like(xs, -8.0)]))

    # East hairpin: centre (18, 0), radius 8
    points.append(arc(18, 0, 8, -np.pi / 2, np.pi / 2, 40))

    # Top straight: east (18, 8) → west (-18, 8)
    xs = np.linspace(18, -18, 60, endpoint=False)
    points.append(np.column_stack([xs, np.full_like(xs, 8.0)]))

    # West hairpin: centre (-18, 0), radius 8
    points.append(arc(-18, 0, 8, np.pi / 2, 3 * np.pi / 2, 40))

    base = np.vstack(points)
    tck, _ = splprep([base[:, 0], base[:, 1]], s=0, per=True, k=3)
    u = np.linspace(0, 1, n_gates, endpoint=False)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gates", type=int, default=88,
                        help="Number of gates to generate (default: 88)")
    args = parser.parse_args()

    centre = build_centreline(args.n_gates)
    left, right = offset_path(centre, HALF_WIDTH)

    out = Path(__file__).parent / "track.csv"
    df = pd.DataFrame({
        "left_x":  np.round(left[:, 0], 4),
        "left_y":  np.round(left[:, 1], 4),
        "right_x": np.round(right[:, 0], 4),
        "right_y": np.round(right[:, 1], 4),
    })
    df.to_csv(out, index=False)
    print(f"Generated {len(centre)} gate pairs → {out}")


if __name__ == "__main__":
    main()
