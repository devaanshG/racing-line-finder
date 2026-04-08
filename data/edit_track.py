"""
Interactive track viewer and width editor.

Loads a track CSV, displays it, and lets you adjust the track half-width
via a slider.  The centreline (midpoint of each gate) is kept fixed;
only the left/right cone offsets change.

Usage
-----
    python data/edit_track.py --track data/track.csv

Controls
--------
    Slider      Drag to adjust half-width
    S key       Save current width back to the CSV (overwrites)
    R key       Reset to original width
    Escape      Quit without saving
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button


# ---------------------------------------------------------------------------
# Track geometry helpers
# ---------------------------------------------------------------------------

def load_track(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    left  = df[["left_x",  "left_y"]].to_numpy(dtype=float)
    right = df[["right_x", "right_y"]].to_numpy(dtype=float)
    return left, right


def centreline_and_normals(left: np.ndarray,
                            right: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns
    -------
    centre  : (N, 2) midpoint of each gate
    normal  : (N, 2) unit vector pointing from centre → right cone
    init_hw : float  mean half-width of the original track
    """
    centre   = (left + right) / 2.0
    half_vec = right - centre                             # (N, 2)
    magnitudes = np.linalg.norm(half_vec, axis=1, keepdims=True)
    magnitudes = np.maximum(magnitudes, 1e-9)
    normal   = half_vec / magnitudes
    init_hw  = float(np.mean(np.linalg.norm(half_vec, axis=1)))
    return centre, normal, init_hw


def apply_width(centre: np.ndarray, normal: np.ndarray,
                half_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Recompute left/right cones from centreline + normals + half_width."""
    right = centre + half_width * normal
    left  = centre - half_width * normal
    return left, right


def save_track(path: str, left: np.ndarray, right: np.ndarray) -> None:
    df = pd.DataFrame({
        "left_x":  np.round(left[:,  0], 4),
        "left_y":  np.round(left[:,  1], 4),
        "right_x": np.round(right[:, 0], 4),
        "right_y": np.round(right[:, 1], 4),
    })
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Interactive editor
# ---------------------------------------------------------------------------

_LEFT_COLOUR  = "#4499ff"
_RIGHT_COLOUR = "#f0c040"
_CENTRE_COLOUR = "#ffffff"
_GATE_COLOUR  = "#555555"


def run_editor(track_path: str) -> None:
    left_orig, right_orig = load_track(track_path)
    centre, normal, init_hw = centreline_and_normals(left_orig, right_orig)
    n_gates = len(centre)

    # Infer units label from magnitude (rough heuristic: >20 → probably pixels)
    unit_label = "px" if init_hw > 5 else "m"

    # ---- Figure layout ----
    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#1a1a1a")

    # Main track axes — leave room at bottom for slider + buttons
    ax = fig.add_axes([0.05, 0.18, 0.9, 0.76])
    ax.set_facecolor("#111111")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # ---- Initial plot objects ----
    left0, right0 = apply_width(centre, normal, init_hw)

    def _closed(arr):
        return np.vstack([arr, arr[0]])

    lc = _closed(left0)
    rc = _closed(right0)
    cc = _closed(centre)

    line_left,  = ax.plot(lc[:, 0], lc[:, 1], color=_LEFT_COLOUR,
                          lw=1.5, label="Left boundary")
    line_right, = ax.plot(rc[:, 0], rc[:, 1], color=_RIGHT_COLOUR,
                          lw=1.5, label="Right boundary")
    line_centre, = ax.plot(cc[:, 0], cc[:, 1], color=_CENTRE_COLOUR,
                           lw=0.8, alpha=0.35, linestyle="--", label="Centreline")

    dots_left  = ax.scatter(left0[:,  0], left0[:,  1],
                            c=_LEFT_COLOUR,  s=12, zorder=4)
    dots_right = ax.scatter(right0[:, 0], right0[:, 1],
                            c=_RIGHT_COLOUR, s=12, zorder=4)

    # Gate lines (draw every Nth to avoid clutter)
    step = max(1, n_gates // 60)
    gate_lines = []
    for l, r in zip(left0[::step], right0[::step]):
        ln, = ax.plot([l[0], r[0]], [l[1], r[1]],
                      color=_GATE_COLOUR, lw=0.6, alpha=0.6)
        gate_lines.append(ln)

    # Start marker
    start_dot = ax.scatter(*centre[0], c="lime", s=120, zorder=6, marker="*")

    ax.legend(
        handles=[
            mpatches.Patch(color=_LEFT_COLOUR,   label="Left"),
            mpatches.Patch(color=_RIGHT_COLOUR,  label="Right"),
            mpatches.Patch(color=_CENTRE_COLOUR, label="Centreline", alpha=0.4),
        ],
        loc="upper right", fontsize=9, facecolor="#222222", edgecolor="#555555",
        labelcolor="white",
    )

    title = ax.set_title(
        f"{Path(track_path).name}  |  {n_gates} gates  |  "
        f"half-width = {init_hw:.1f} {unit_label}",
        color="white", fontsize=11,
    )

    # ---- Slider ----
    ax_slider = fig.add_axes([0.15, 0.09, 0.6, 0.03])
    ax_slider.set_facecolor("#222222")
    slider_max = init_hw * 3.0
    slider_min = max(1.0, init_hw * 0.1)
    slider = Slider(
        ax_slider, f"Half-width ({unit_label})",
        slider_min, slider_max,
        valinit=init_hw,
        color=_RIGHT_COLOUR,
    )
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    # ---- Buttons ----
    ax_save  = fig.add_axes([0.80, 0.04, 0.09, 0.04])
    ax_reset = fig.add_axes([0.68, 0.04, 0.09, 0.04])
    btn_save  = Button(ax_save,  "Save (S)",
                       color="#2a5c2a", hovercolor="#3a8c3a")
    btn_reset = Button(ax_reset, "Reset (R)",
                       color="#444444", hovercolor="#666666")
    for btn in (btn_save, btn_reset):
        btn.label.set_color("white")

    status_text = fig.text(
        0.15, 0.03, "",
        color="#aaaaaa", fontsize=9, va="bottom",
    )

    # ---- Update callback ----
    def _update(hw: float) -> None:
        l, r = apply_width(centre, normal, hw)
        lc2, rc2 = _closed(l), _closed(r)

        line_left.set_data(lc2[:, 0], lc2[:, 1])
        line_right.set_data(rc2[:, 0], rc2[:, 1])
        dots_left.set_offsets(l)
        dots_right.set_offsets(r)

        for i, ln in enumerate(gate_lines):
            gi = i * step
            ln.set_data([l[gi, 0], r[gi, 0]], [l[gi, 1], r[gi, 1]])

        title.set_text(
            f"{Path(track_path).name}  |  {n_gates} gates  |  "
            f"half-width = {hw:.1f} {unit_label}  "
            f"(track width = {hw * 2:.1f} {unit_label})"
        )
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    # ---- Save / reset ----
    def _save(_=None) -> None:
        hw = slider.val
        l, r = apply_width(centre, normal, hw)
        save_track(track_path, l, r)
        status_text.set_text(
            f"Saved  →  {track_path}  "
            f"(half-width = {hw:.2f} {unit_label})"
        )
        fig.canvas.draw_idle()
        print(f"Saved: half-width = {hw:.2f} {unit_label}  →  {track_path}")

    def _reset(_=None) -> None:
        slider.set_val(init_hw)
        status_text.set_text("Reset to original width.")
        fig.canvas.draw_idle()

    btn_save.on_clicked(_save)
    btn_reset.on_clicked(_reset)

    def _on_key(event) -> None:
        if event.key == "s":
            _save()
        elif event.key == "r":
            _reset()
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualise and edit track width interactively.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--track", default="data/track.csv",
                   help="Path to track CSV (left_x,left_y,right_x,right_y)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not Path(args.track).exists():
        print(f"Error: track file not found: {args.track}", file=sys.stderr)
        sys.exit(1)
    run_editor(args.track)


if __name__ == "__main__":
    main()
