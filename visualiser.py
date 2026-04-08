"""
Visualiser module.

Responsibilities:
- All matplotlib plotting lives here; no plt calls in other modules.
- plot_track      : cone boundaries, gate lines, start marker.
- plot_centerline : overlay the naive gate-midpoint centerline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


# --- Colours consistent across all plots ---
_LEFT_COLOUR       = "#1f77b4"   # blue   (left boundary)
_RIGHT_COLOUR      = "#f0c040"   # yellow (right boundary)
_GATE_COLOUR       = "#aaaaaa"   # grey   (gate lines)
_START_COLOUR      = "#2ca02c"   # green  (start marker)
_CENTRELINE_COLOUR = "#ff7f0e"   # orange (naive centerline)


def _closed(arr: np.ndarray) -> np.ndarray:
    """Append the first row so a boundary line closes the loop."""
    return np.vstack([arr, arr[0]])


def _heading(centre: np.ndarray) -> np.ndarray:
    """Unit vector from gate 0 midpoint toward gate 1 midpoint."""
    h = centre[1] - centre[0]
    n = np.linalg.norm(h)
    return h / n if n > 1e-9 else np.array([1.0, 0.0])


def _arrow_scale(left_cones: np.ndarray, right_cones: np.ndarray) -> float:
    """Scale start arrow to ~2 gate widths."""
    widths = np.linalg.norm(right_cones - left_cones, axis=1)
    return float(np.mean(widths)) * 1.5


def plot_track(
    left_cones: np.ndarray,
    right_cones: np.ndarray,
    centerline: np.ndarray | None = None,
    closed: bool = True,
    title: str = "Track Map",
) -> None:
    """
    Plot cone boundaries, gate lines, start marker, and optionally the
    naive gate-midpoint centerline.

    Parameters
    ----------
    left_cones, right_cones : np.ndarray (N, 2)
    centerline              : np.ndarray (N, 2) or None
        If provided, the gate-midpoint centerline is overlaid.
    closed : bool
        If True, boundary lines and centerline close the loop.
    title : str
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    def maybe_close(arr):
        return _closed(arr) if closed else arr

    # Boundary lines
    lc = maybe_close(left_cones)
    rc = maybe_close(right_cones)
    ax.plot(lc[:, 0], lc[:, 1], color=_LEFT_COLOUR,  linewidth=1.2, alpha=0.7)
    ax.plot(rc[:, 0], rc[:, 1], color=_RIGHT_COLOUR, linewidth=1.2, alpha=0.9)

    # Gate lines (thin, every gate)
    for l, r in zip(left_cones, right_cones):
        ax.plot([l[0], r[0]], [l[1], r[1]],
                color=_GATE_COLOUR, linewidth=0.5, alpha=0.35)

    # Cone markers
    ax.scatter(left_cones[:, 0],  left_cones[:, 1],
               c=_LEFT_COLOUR,  s=22, zorder=3)
    ax.scatter(right_cones[:, 0], right_cones[:, 1],
               c=_RIGHT_COLOUR, edgecolors="black", linewidths=0.4, s=22, zorder=3)

    # Centerline overlay
    if centerline is not None:
        cc = maybe_close(centerline)
        ax.plot(cc[:, 0], cc[:, 1],
                color=_CENTRELINE_COLOUR, linewidth=1.8,
                linestyle="--", alpha=0.85, zorder=4)
        ax.scatter(centerline[:, 0], centerline[:, 1],
                   c=_CENTRELINE_COLOUR, s=14, zorder=5, alpha=0.7)

    # Start marker + heading arrow
    start = (left_cones[0] + right_cones[0]) / 2.0
    h = _heading(centerline if centerline is not None
                 else (left_cones + right_cones) / 2.0)
    scale = _arrow_scale(left_cones, right_cones)

    ax.scatter(*start, c=_START_COLOUR, s=130, zorder=6, marker="*")
    ax.annotate(
        "", xy=start + h * scale, xytext=start,
        arrowprops=dict(arrowstyle="->", color=_START_COLOUR, lw=2.0),
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=_LEFT_COLOUR,  label="Left cones"),
        mpatches.Patch(color=_RIGHT_COLOUR, label="Right cones"),
        mlines.Line2D([], [], color=_START_COLOUR, marker="*",
                      markersize=10, label="Start / heading", linestyle="None"),
    ]
    if centerline is not None:
        legend_handles.append(
            mlines.Line2D([], [], color=_CENTRELINE_COLOUR,
                          linewidth=1.8, linestyle="--", label="Centerline")
        )

    ax.set_aspect("equal")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.show()
