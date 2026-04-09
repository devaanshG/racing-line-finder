"""
Visualiser module.

Responsibilities:
- All matplotlib plotting lives here; no plt calls in other modules.
- plot_track : cone boundaries, gate lines, start marker,
               optional centerline and DP path overlays.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection


# --- Colours consistent across all plots ---
_LEFT_COLOUR       = "#1f77b4"   # blue   (left boundary)
_RIGHT_COLOUR      = "#f0c040"   # yellow (right boundary)
_GATE_COLOUR       = "#aaaaaa"   # grey   (gate lines)
_START_COLOUR      = "#2ca02c"   # green  (start marker)
_CENTRELINE_COLOUR = "#ff7f0e"   # orange (naive centerline)
_PATH_COLOUR       = "#e84040"   # red    (DP path — solid fallback, no speed data)
_SPEED_CMAP        = "plasma"    # colormap for speed-coloured path


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
    left_cones:    np.ndarray,
    right_cones:   np.ndarray,
    centerline:    np.ndarray | None = None,
    dp_path:       np.ndarray | None = None,
    speed_profile: np.ndarray | None = None,
    closed:        bool = True,
    title:         str  = "Track Map",
) -> None:
    """
    Plot cone boundaries, gate lines, start marker, and optional overlays.

    Parameters
    ----------
    left_cones, right_cones : np.ndarray (N, 2)
    centerline    : np.ndarray (N, 2) or None — naive gate-midpoint centerline
    dp_path       : np.ndarray (N, 2) or None — DP-optimised raw path
    speed_profile : np.ndarray (N,)   or None — speed in m/s at each gate;
                    when provided alongside dp_path the path is coloured by
                    speed using the plasma colormap
    closed        : bool — close boundary lines and paths into a loop
    title         : str
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
                color=_CENTRELINE_COLOUR, linewidth=1.5,
                linestyle="--", alpha=0.6, zorder=4)

    # DP path overlay
    if dp_path is not None:
        pp = maybe_close(dp_path)
        if speed_profile is not None:
            # Colour the path by speed using the plasma colormap
            sp = np.append(speed_profile, speed_profile[0]) if closed else speed_profile
            segments = np.stack([pp[:-1], pp[1:]], axis=1)          # (N, 2, 2)
            seg_speeds = (sp[:-1] + sp[1:]) / 2.0                   # midpoint speed per segment
            norm = plt.Normalize(vmin=sp.min(), vmax=sp.max())
            lc = LineCollection(segments, cmap=_SPEED_CMAP, norm=norm,
                                linewidth=2.2, zorder=6, alpha=0.95)
            lc.set_array(seg_speeds)
            ax.add_collection(lc)
            fig.colorbar(lc, ax=ax, label="Speed (m/s)", fraction=0.03, pad=0.04)
        else:
            ax.plot(pp[:, 0], pp[:, 1],
                    color=_PATH_COLOUR, linewidth=2.2, alpha=0.95, zorder=6)
        ax.scatter(dp_path[:, 0], dp_path[:, 1],
                   c=_PATH_COLOUR, s=16, zorder=7, alpha=0.85)

    # Start marker + heading arrow
    ref = dp_path if dp_path is not None else (
          centerline if centerline is not None else
          (left_cones + right_cones) / 2.0)
    start = (left_cones[0] + right_cones[0]) / 2.0
    h = _heading(ref)
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
                          linewidth=1.5, linestyle="--", label="Centerline", alpha=0.6)
        )
    if dp_path is not None:
        label = "DP path (speed)" if speed_profile is not None else "DP path"
        legend_handles.append(
            mlines.Line2D([], [], color=_PATH_COLOUR,
                          linewidth=2.2, label=label)
        )

    ax.set_aspect("equal")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.show()
