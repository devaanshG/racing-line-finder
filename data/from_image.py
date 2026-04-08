"""
Digitise a top-down track map image into track.csv.

Usage
-----
Manual mode (default) — click the centreline in driving order:
    python data/from_image.py --image track.png

    Controls:
        Left-click   — place a centreline point
        Right-click  — undo last point
        Enter        — finish and run pipeline

Automated outline mode — B&W line drawing:
    python data/from_image.py --image track.png --mode outline

Automated ribbon mode — filled colour band:
    python data/from_image.py --image track.png --mode ribbon

Options
-------
--image           Path to track image (PNG, JPG, …)
--mode            manual | outline | ribbon  (default: manual)
--half-width-px   Pixels from centreline to each cone row (default: 8)
--n-gates         Number of gate pairs to output (default: 100)
--no-loop         Treat track as open (not closed). Default: closed loop.
--track-length-m  Known centreline length in metres for coordinate scaling.
                  If omitted, output is in pixels.
--out-dir         Output directory (default: data/)
--show            Preview final gates before saving
--debug           [outline/ribbon] Show raw colour mask for diagnosis

Automated-mode only
-------------------
--track-color     R,G,B  (omit to click interactively)
--tolerance       Colour-match tolerance 0–255  (default: 40)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.interpolate import splprep, splev


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load image as float32 RGB (H, W, 3) with values in [0, 255]."""
    img = mpimg.imread(path)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.float32)
    else:
        img = img.astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


# ---------------------------------------------------------------------------
# Manual mode: interactive centreline clicking
# ---------------------------------------------------------------------------

def click_centreline(img: np.ndarray, closed: bool) -> np.ndarray:
    """
    Show the image and let the user click centreline points in driving order.

    Controls
    --------
    Left-click   place a point
    Right-click  undo last point
    Enter        finish

    Returns
    -------
    pts : np.ndarray, shape (N, 2)  — (x, y) pixel coordinates
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img.astype(np.uint8))
    loop_str = "closed loop" if closed else "open path"
    ax.set_title(
        f"Click centreline in DRIVING ORDER ({loop_str}).\n"
        "Left-click = place point   Right-click = undo   Enter = done",
        fontsize=11,
    )
    ax.axis("off")

    points: list[list[float]] = []
    dots = ax.scatter([], [], c="lime", s=20, zorder=5)
    line, = ax.plot([], [], color="lime", lw=1.2, alpha=0.8)
    count_text = ax.text(
        0.01, 0.99, "Points: 0",
        transform=ax.transAxes, va="top", fontsize=10,
        color="white", bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
    )

    def _redraw():
        if points:
            arr = np.array(points)
            dots.set_offsets(arr)
            xs = np.append(arr[:, 0], arr[0, 0]) if closed and len(arr) > 2 else arr[:, 0]
            ys = np.append(arr[:, 1], arr[0, 1]) if closed and len(arr) > 2 else arr[:, 1]
            line.set_data(xs, ys)
        else:
            dots.set_offsets(np.empty((0, 2)))
            line.set_data([], [])
        count_text.set_text(f"Points: {len(points)}")
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if event.button == 1:
            points.append([event.xdata, event.ydata])
            _redraw()
        elif event.button == 3 and points:
            points.pop()
            _redraw()

    def _on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.tight_layout()
    plt.show(block=True)

    if len(points) < 4:
        raise ValueError(f"Need at least 4 points; got {len(points)}.")
    return np.array(points)


# ---------------------------------------------------------------------------
# Spline fitting and resampling
# ---------------------------------------------------------------------------

def fit_spline(pts_xy: np.ndarray, closed: bool):
    """
    Fit a parametric cubic spline through (x, y) pixel points.
    Returns (tck, u) from scipy splprep.
    """
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    if closed:
        # splprep with per=True requires the path NOT to repeat the first point
        tck, u = splprep([x, y], s=0, per=True, k=3)
    else:
        tck, u = splprep([x, y], s=0, per=False, k=3)
    return tck, u


def resample_spline(tck, n: int, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the spline at n evenly-spaced parameter values.

    Returns
    -------
    pts      : (n, 2) resampled centreline points  (x, y)
    tangents : (n, 2) spline tangent vectors at each point
    """
    u_new = np.linspace(0.0, 1.0, n, endpoint=not closed)
    xy  = np.array(splev(u_new, tck)).T       # (n, 2)
    dxy = np.array(splev(u_new, tck, der=1)).T  # (n, 2) tangent
    return xy, dxy


def normals_from_tangents(tangents: np.ndarray) -> np.ndarray:
    """
    Left-pointing unit normals (90° CCW from each tangent).
    In screen/pixel space where y increases downward, this puts
    the left cone to the left of the direction of travel.
    """
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    t = tangents / norms
    return np.column_stack([-t[:, 1], t[:, 0]])


def make_gates(centre_xy: np.ndarray, tangents: np.ndarray,
               half_width: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Place left and right cones at ±half_width pixels from the centreline,
    perpendicular to the driving direction.

    Returns left_xy, right_xy  — both (n, 2) in pixel (x, y) coords.
    """
    normals = normals_from_tangents(tangents)
    left_xy  = centre_xy + half_width * normals
    right_xy = centre_xy - half_width * normals
    return left_xy, right_xy


# ---------------------------------------------------------------------------
# Automated detection (outline / ribbon) — kept for completeness
# ---------------------------------------------------------------------------

def pick_color_interactively(img: np.ndarray) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img.astype(np.uint8))
    ax.set_title("Click ONCE on the track surface, then close", fontsize=12)
    ax.axis("off")
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not pts:
        raise RuntimeError("No point selected.")
    x, y = int(round(pts[0][0])), int(round(pts[0][1]))
    colour = img[y, x, :3]
    print(f"  Sampled colour at ({x}, {y}): RGB = {colour.astype(int)}")
    return colour


def _color_mask(img, track_rgb, tolerance):
    diff = img[:, :, :3].astype(float) - track_rgb.astype(float)
    return np.linalg.norm(diff, axis=-1) < tolerance


def _largest_component(mask):
    labeled, n = ndimage.label(mask)
    if n == 0:
        raise ValueError("No connected component found in mask.")
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return (labeled == (np.argmax(sizes) + 1)).astype(bool)


def build_mask_outline(img, track_rgb, tolerance):
    mask = _color_mask(img, track_rgb, tolerance)
    return ndimage.binary_closing(mask, iterations=2).astype(bool)


def build_mask_ribbon(img, track_rgb, tolerance):
    mask = _color_mask(img, track_rgb, tolerance)
    mask = ndimage.binary_opening(mask, iterations=1)
    return ndimage.binary_closing(mask, iterations=3).astype(bool)


def extract_ribbon_edges(mask):
    track  = _largest_component(mask)
    filled = ndimage.binary_fill_holes(track)
    infield_all = filled & ~track
    if not infield_all.any():
        raise ValueError("No infield detected. Use --mode outline or --mode manual.")
    infield = _largest_component(infield_all)
    outer = track & ndimage.binary_dilation(~filled)
    inner = track & ndimage.binary_dilation(infield)
    return outer, inner


def extract_centreline_mask(mask):
    track   = _largest_component(mask)
    thinned = ndimage.binary_erosion(track, iterations=1)
    return _largest_component(thinned if thinned.any() else track)


def _walk_boundary(edge):
    pts    = np.argwhere(edge)
    if len(pts) == 0:
        raise ValueError("Empty edge.")
    pt_set = {(int(r), int(c)) for r, c in pts}
    start  = tuple(pts[np.lexsort((pts[:, 1], pts[:, 0]))[0]])
    ordered = [start]
    pt_set.discard(start)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while pt_set:
        r, c = ordered[-1]
        found = None
        for dr, dc in offsets:
            nb = (r + dr, c + dc)
            if nb in pt_set:
                found = nb
                break
        if found is None:
            remaining = np.array(list(pt_set))
            dists = np.sum((remaining - np.array([r, c])) ** 2, axis=1)
            ni = int(np.argmin(dists))
            if dists[ni] > 9:
                print(f"  Warning: walk stopped early ({len(ordered)} pts traced).",
                      file=sys.stderr)
                break
            found = tuple(remaining[ni])
        ordered.append(found)
        pt_set.discard(found)
    return np.array(ordered)


def _signed_area(pts):
    x, y = pts[:, 1], -pts[:, 0]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_ccw(pts):
    return pts if _signed_area(pts) >= 0 else pts[::-1]


def _align_start(ref, arr):
    return np.roll(arr, -int(np.argmin(np.sum((arr - ref[0]) ** 2, axis=1))), axis=0)


def _resample_path(pts, n):
    closed = np.vstack([pts, pts[0]])
    seg    = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum    = np.concatenate([[0.0], np.cumsum(seg)])
    total  = cum[-1]
    result = np.zeros((n, 2))
    for i, t in enumerate(np.linspace(0.0, total, n, endpoint=False)):
        idx = np.clip(int(np.searchsorted(cum, t, side="right")) - 1, 0, len(pts)-1)
        nxt = (idx + 1) % len(pts)
        frac = (t - cum[idx]) / seg[idx] if seg[idx] > 1e-9 else 0.0
        result[i] = pts[idx] + frac * (pts[nxt] - pts[idx])
    return result


# ---------------------------------------------------------------------------
# Debug / preview
# ---------------------------------------------------------------------------

def debug_mask(img, mask, track_rgb, tolerance):
    overlay = np.zeros((*img.shape[:2], 4), dtype=np.uint8)
    overlay[mask] = [255, 0, 0, 180]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(img.astype(np.uint8))
    axes[0].imshow(overlay)
    axes[0].set_title(
        f"Red = detected pixels\n"
        f"colour={track_rgb.astype(int)}  tolerance={tolerance:.0f}  "
        f"matched={mask.sum():,} ({mask.sum()/mask.size:.1%})",
        fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray", interpolation="nearest")
    axes[1].set_title("Raw mask (white = detected)", fontsize=10)
    axes[1].axis("off")
    fig.suptitle("DEBUG — close to continue", fontsize=11, color="darkred")
    plt.tight_layout()
    plt.show()


def preview_gates(img: np.ndarray, left_xy: np.ndarray,
                  right_xy: np.ndarray, centre_xy: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left panel: overlaid on the source image
    axes[0].imshow(img.astype(np.uint8))
    axes[0].plot(right_xy[:, 0],  right_xy[:, 1],  "y-",    lw=1.2, label="Right cones")
    axes[0].plot(left_xy[:, 0],   left_xy[:, 1],   color="#4499ff", lw=1.2, label="Left cones")
    axes[0].plot(centre_xy[:, 0], centre_xy[:, 1], "w--",   lw=0.8, alpha=0.5, label="Centreline")
    # Gate lines
    for l, r in zip(left_xy[::max(1, len(left_xy)//40)],
                    right_xy[::max(1, len(right_xy)//40)]):
        axes[0].plot([l[0], r[0]], [l[1], r[1]], color="gray", lw=0.5, alpha=0.5)
    axes[0].scatter(*right_xy[0], c="lime", s=80, zorder=5, label="Gate 0")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("Gates overlaid on image")
    axes[0].axis("off")

    # Right panel: clean coordinate space (y-flipped for standard orientation)
    axes[1].plot(right_xy[:, 0],  -right_xy[:, 1],  "y-",    lw=1.5, label="Right")
    axes[1].plot(left_xy[:, 0],   -left_xy[:, 1],   color="#4499ff", lw=1.5, label="Left")
    axes[1].plot(centre_xy[:, 0], -centre_xy[:, 1], "w--",   lw=0.8, alpha=0.5)
    axes[1].scatter(right_xy[0, 0], -right_xy[0, 1], c="lime", s=80, zorder=5, label="Gate 0")
    axes[1].set_aspect("equal")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Extracted track (pixel coords)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor("#111111")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def scale_to_metres(xy: np.ndarray, ref_xy: np.ndarray,
                    img_h: int, track_length_m: float) -> np.ndarray:
    """Scale pixel (x, y) → metre (x, y) using ref_xy arc length."""
    closed = np.vstack([ref_xy, ref_xy[0]])
    px_len = np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1))
    scale  = track_length_m / px_len
    # Flip y so increasing y = up (standard map orientation)
    out = xy.copy()
    out[:, 1] = img_h - out[:, 1]
    return out * scale


def to_xy_flipped(xy: np.ndarray, img_h: int) -> np.ndarray:
    out = xy.copy()
    out[:, 1] = img_h - out[:, 1]
    return out


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Digitise a track map image into cone CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",    required=True, help="Path to the track image")
    p.add_argument("--mode",     default="manual",
                   choices=["manual", "outline", "ribbon"],
                   help="manual=click centreline, outline/ribbon=auto-detect")
    p.add_argument("--half-width-px", type=float, default=8.0,
                   help="Pixels from centreline to each cone")
    p.add_argument("--n-gates",       type=int,   default=100,
                   help="Number of gate pairs to output")
    p.add_argument("--no-loop",       action="store_true",
                   help="Treat track as open path (default: closed loop)")
    p.add_argument("--track-length-m", type=float, default=None,
                   help="Centreline length in metres for coordinate scaling")
    p.add_argument("--out-dir",  default="data",
                   help="Output directory for CSVs")
    p.add_argument("--show",     action="store_true",
                   help="Preview gates before saving")
    # Automated-mode options
    p.add_argument("--track-color", default=None, metavar="R,G,B",
                   help="[outline/ribbon] Track colour. Omit to click.")
    p.add_argument("--tolerance",   type=float, default=40.0,
                   help="[outline/ribbon] Colour-match tolerance (L2, 0–255)")
    p.add_argument("--debug",   action="store_true",
                   help="[outline/ribbon] Show raw colour mask for diagnosis")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    closed = not args.no_loop

    print(f"Loading image: {args.image}")
    img = load_image(args.image)
    print(f"Image: {img.shape[1]} × {img.shape[0]} px  |  mode: {args.mode}")

    # ------------------------------------------------------------------
    # Manual mode
    # ------------------------------------------------------------------
    if args.mode == "manual":
        clicked_xy = click_centreline(img, closed)
        print(f"  {len(clicked_xy)} centreline points clicked.")

        tck, _ = fit_spline(clicked_xy, closed)
        centre_xy, tangents = resample_spline(tck, args.n_gates, closed)
        left_xy, right_xy   = make_gates(centre_xy, tangents, args.half_width_px)

    # ------------------------------------------------------------------
    # Automated modes (outline / ribbon)
    # ------------------------------------------------------------------
    else:
        if args.track_color:
            parts = [float(v) for v in args.track_color.split(",")]
            if len(parts) != 3:
                raise ValueError("--track-color must be 'R,G,B'")
            track_rgb = np.array(parts, dtype=float)
        else:
            track_rgb = pick_color_interactively(img)

        print(f"Building mask (tolerance={args.tolerance}) …")
        mask = (build_mask_outline if args.mode == "outline"
                else build_mask_ribbon)(img, track_rgb, args.tolerance)
        frac = mask.sum() / mask.size
        print(f"  Matched: {mask.sum():,} px ({frac:.1%})")
        if frac < 0.002:
            print("  Warning: very few pixels — try --tolerance 80", file=sys.stderr)
        if frac > 0.5:
            print("  Warning: >50% matched — try lower --tolerance", file=sys.stderr)
        if args.debug:
            debug_mask(img, mask, track_rgb, args.tolerance)

        if args.mode == "ribbon":
            outer_edge, inner_edge = extract_ribbon_edges(mask)
            outer_rc = _ensure_ccw(_walk_boundary(outer_edge))
            inner_rc = _ensure_ccw(_walk_boundary(inner_edge))
            inner_rc = _align_start(outer_rc, inner_rc)
            right_rc = _resample_path(outer_rc, args.n_gates)
            left_rc  = _resample_path(inner_rc, args.n_gates)
            centre_rc = (right_rc + left_rc) / 2.0
            # (row, col) → (x, y)
            def rc_to_xy(rc): return np.column_stack([rc[:, 1], rc[:, 0]])
            right_xy  = rc_to_xy(right_rc)
            left_xy   = rc_to_xy(left_rc)
            centre_xy = rc_to_xy(centre_rc)

        else:  # outline
            centre_mask = extract_centreline_mask(mask)
            centre_rc   = _ensure_ccw(_walk_boundary(centre_mask))
            centre_rc   = _resample_path(centre_rc, args.n_gates)
            centre_xy   = np.column_stack([centre_rc[:, 1], centre_rc[:, 0]])
            # Compute tangents from finite differences on the resampled path
            tangents    = np.gradient(centre_xy, axis=0)
            left_xy, right_xy = make_gates(centre_xy, tangents, args.half_width_px)

    # ------------------------------------------------------------------
    # Optional preview
    # ------------------------------------------------------------------
    if args.show:
        preview_gates(img, left_xy, right_xy, centre_xy)

    # ------------------------------------------------------------------
    # Convert coordinates and save
    # ------------------------------------------------------------------
    ref = centre_xy
    if args.track_length_m:
        left_out  = scale_to_metres(left_xy,  ref, img.shape[0], args.track_length_m)
        right_out = scale_to_metres(right_xy, ref, img.shape[0], args.track_length_m)
        unit = "m"
    else:
        left_out  = to_xy_flipped(left_xy,  img.shape[0])
        right_out = to_xy_flipped(right_xy, img.shape[0])
        unit = "px"

    out_dir    = Path(args.out_dir)
    track_path = out_dir / "track.csv"
    df = pd.DataFrame({
        "left_x":  np.round(left_out[:, 0], 4),
        "left_y":  np.round(left_out[:, 1], 4),
        "right_x": np.round(right_out[:, 0], 4),
        "right_y": np.round(right_out[:, 1], 4),
    })
    df.to_csv(track_path, index=False)
    print(f"Saved {args.n_gates} gate pairs ({unit}) → {track_path}")


if __name__ == "__main__":
    main()
