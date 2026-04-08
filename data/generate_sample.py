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


def build_centreline() -> np.ndarray:
    """
    Assemble a closed centreline from straight and arc segments.
    Gate density: roughly one gate every 1.5 m along each segment.
    """
    points = []

    # --- Bottom straight: west (-18, -8) → east (18, -8) ---
    xs = np.linspace(-18, 18, 26, endpoint=False)
    points.append(np.column_stack([xs, np.full_like(xs, -8.0)]))

    # --- East hairpin: centre (18, 0), radius 8, from -π/2 → +π/2 ---
    points.append(arc(18, 0, 8, -np.pi / 2, np.pi / 2, 18))

    # --- Top straight: east (18, 8) → west (-18, 8) ---
    xs = np.linspace(18, -18, 26, endpoint=False)
    points.append(np.column_stack([xs, np.full_like(xs, 8.0)]))

    # --- West hairpin: centre (-18, 0), radius 8, from +π/2 → +3π/2 ---
    points.append(arc(-18, 0, 8, np.pi / 2, 3 * np.pi / 2, 18))

    return np.vstack(points)


def main():
    centre = build_centreline()
    left, right = offset_path(centre, HALF_WIDTH)

    out = Path(__file__).parent
    pd.DataFrame(left, columns=["x", "y"]).round(4).to_csv(out / "left_cones.csv", index=False)
    pd.DataFrame(right, columns=["x", "y"]).round(4).to_csv(out / "right_cones.csv", index=False)
    print(f"Generated {len(centre)} gate pairs → left_cones.csv, right_cones.csv")


if __name__ == "__main__":
    main()
