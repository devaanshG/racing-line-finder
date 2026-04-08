"""
Visualiser module.

Responsibilities:
- All matplotlib plotting lives here; no plt calls in other modules.
- plot_track : cone boundaries, gate lines, start marker.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# --- Colours consistent across all plots ---
_LEFT_COLOUR = "#1f77b4"   # blue  (left / inner boundary)
_RIGHT_COLOUR = "#f0c040"  # yellow (right / outer boundary)
_GATE_COLOUR = "#aaaaaa"
_START_COLOUR = "#2ca02c"  # green


def _closed(arr: np.ndarray) -> np.ndarray:
    """Append the first row to the end so a boundary line closes the loop."""
    return np.vstack([arr, arr[0]])


def plot_track(
    left_cones: np.ndarray,
    right_cones: np.ndarray,
    title: str = "Track Map — Stage 1",
) -> None:
    """
    Plot cone boundaries, gate lines, and start position.

    Parameters
    ----------
    left_cones, right_cones : np.ndarray of shape (N, 2)
    title : str
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Boundary lines (closed loop)
    lc = _closed(left_cones)
    rc = _closed(right_cones)
    ax.plot(lc[:, 0], lc[:, 1], color=_LEFT_COLOUR, linewidth=1.2, alpha=0.6)
    ax.plot(rc[:, 0], rc[:, 1], color=_RIGHT_COLOUR, linewidth=1.2, alpha=0.8)

    # Gate lines
    for l, r in zip(left_cones, right_cones):
        ax.plot([l[0], r[0]], [l[1], r[1]], color=_GATE_COLOUR, linewidth=0.5, alpha=0.4)

    # Cone markers
    ax.scatter(
        left_cones[:, 0], left_cones[:, 1],
        c=_LEFT_COLOUR, s=25, zorder=3, label="Left cones"
    )
    ax.scatter(
        right_cones[:, 0], right_cones[:, 1],
        c=_RIGHT_COLOUR, edgecolors="black", linewidths=0.4, s=25, zorder=3, label="Right cones"
    )

    # Start position: midpoint of gate 0 with a direction arrow
    start = (left_cones[0] + right_cones[0]) / 2.0
    heading = (left_cones[1] + right_cones[1]) / 2.0 - start
    heading /= np.linalg.norm(heading)

    ax.scatter(*start, c=_START_COLOUR, s=120, zorder=5, marker="*")
    ax.annotate(
        "", xy=start + heading * 2.5, xytext=start,
        arrowprops=dict(arrowstyle="->", color=_START_COLOUR, lw=1.8),
    )

    # Legend entry for start
    start_patch = mpatches.Patch(color=_START_COLOUR, label="Start / heading")

    ax.set_aspect("equal")
    ax.legend(handles=[
        mpatches.Patch(color=_LEFT_COLOUR, label="Left cones"),
        mpatches.Patch(color=_RIGHT_COLOUR, label="Right cones"),
        start_patch,
    ], loc="upper right")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.show()
