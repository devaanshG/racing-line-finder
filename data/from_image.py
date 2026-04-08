"""
Digitise a top-down track map image into left_cones.csv / right_cones.csv.

Usage
-----
Outline mode — B&W line drawing (track is a single drawn line):
    python data/from_image.py --image track.png --mode outline

Ribbon mode — filled track band (track surface is a distinct colour):
    python data/from_image.py --image track.png --mode ribbon

With explicit track colour (skips the interactive click):
    python data/from_image.py --image track.png --mode outline --track-color 0,0,0

With real-world scaling:
    python data/from_image.py --image track.png --mode outline --track-length-m 5300

Modes
-----
outline (default)
    For images where the track is drawn as a line or outline on a plain
    background (typical B&W circuit diagrams, hand-drawn sketches).
    The detected line is treated as the CENTRELINE; left/right cone positions
    are offset by --half-width-px either side.
    No aggressive morphology — thin lines are preserved.

ribbon
    For images where the track surface is a visually distinct filled band
    (e.g. coloured race-circuit maps).  Extracts separate inner and outer
    boundary edges of the band.

Options
-------
--image           Path to the track image (PNG, JPG, …)
--mode            'outline' or 'ribbon' (default: outline)
--track-color     Track colour as R,G,B integers 0–255. Omit to pick
                  interactively.
--tolerance       Colour-match tolerance in [0–255] L2 space (default: 40)
--half-width-px   [outline mode] pixels to offset left/right from centreline
                  (default: 8)
--n-gates         Number of gate pairs to output (default: 100)
--track-length-m  Known centreline length in metres for scaling. If omitted,
                  output is in pixels.
--out-dir         Directory to write CSVs (default: data/)
--show            Preview detected boundaries before saving
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load image as float32 RGB array, values in [0, 255]."""
    img = mpimg.imread(path)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.float32)
    else:
        img = img.astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)   # grayscale → RGB
    if img.shape[2] == 4:
        img = img[:, :, :3]                  # RGBA → RGB
    return img


# ---------------------------------------------------------------------------
# Colour picking
# ---------------------------------------------------------------------------

def pick_color_interactively(img: np.ndarray) -> np.ndarray:
    """Open the image; user clicks once on the track; returns sampled RGB."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img.astype(np.uint8))
    ax.set_title("Click ONCE on the track line/surface, then close this window",
                 fontsize=12)
    ax.axis("off")
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not pts:
        raise RuntimeError("No point selected — aborting.")
    x, y = int(round(pts[0][0])), int(round(pts[0][1]))
    colour = img[y, x, :3]
    print(f"  Sampled track colour at ({x}, {y}): RGB = {colour.astype(int)}")
    return colour


# ---------------------------------------------------------------------------
# Mask building
# ---------------------------------------------------------------------------

def _color_distance_mask(img: np.ndarray, track_rgb: np.ndarray,
                          tolerance: float) -> np.ndarray:
    """Bool mask: True where pixel L2 distance from track_rgb < tolerance."""
    diff = img[:, :, :3].astype(float) - track_rgb.astype(float)
    return np.linalg.norm(diff, axis=-1) < tolerance


def build_mask_outline(img: np.ndarray, track_rgb: np.ndarray,
                        tolerance: float) -> np.ndarray:
    """
    Mask for outline mode.  Minimal morphology — only a small closing to
    bridge anti-aliasing gaps.  Thin lines are preserved.
    """
    mask = _color_distance_mask(img, track_rgb, tolerance)
    mask = ndimage.binary_closing(mask, iterations=2)
    return mask.astype(bool)


def build_mask_ribbon(img: np.ndarray, track_rgb: np.ndarray,
                       tolerance: float) -> np.ndarray:
    """
    Mask for ribbon mode.  Opening (iterations=1) to remove speckle, then
    closing to bridge gaps.
    """
    mask = _color_distance_mask(img, track_rgb, tolerance)
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=3)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Largest connected component
# ---------------------------------------------------------------------------

def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n == 0:
        raise ValueError("No connected component found in mask.")
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return (labeled == (np.argmax(sizes) + 1)).astype(bool)


# ---------------------------------------------------------------------------
# Ribbon mode: extract inner/outer edges
# ---------------------------------------------------------------------------

def extract_ribbon_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    From a binary track-ring mask, return outer_edge and inner_edge bool arrays.
    Raises ValueError if no infield can be detected.
    """
    track = _largest_component(mask)
    filled = ndimage.binary_fill_holes(track)

    infield_all = filled & ~track
    if not infield_all.any():
        raise ValueError(
            "No infield detected. Use --mode outline for single-line images."
        )
    infield = _largest_component(infield_all)
    background = ~filled

    outer_edge = track & ndimage.binary_dilation(background)
    inner_edge = track & ndimage.binary_dilation(infield)
    return outer_edge, inner_edge


# ---------------------------------------------------------------------------
# Outline mode: skeletonise and extract single centreline
# ---------------------------------------------------------------------------

def _skeletonise(mask: np.ndarray) -> np.ndarray:
    """
    Thin a binary mask to a 1-px skeleton using iterative erosion.
    (Avoids skimage dependency — uses Zhang-Suen-style thinning via scipy.)
    """
    skel = mask.copy()
    while True:
        eroded = ndimage.binary_erosion(skel)
        opened = ndimage.binary_dilation(eroded)
        # Pixels that would be removed without breaking connectivity
        removable = skel & ~opened
        # Stop when nothing changes
        if not removable.any():
            break
        skel = eroded
    return skel


def extract_centreline_mask(mask: np.ndarray) -> np.ndarray:
    """
    From an outline mask, return the thinned centreline as a bool array.
    Keeps only the largest connected component.
    """
    track = _largest_component(mask)
    # Thin to ~1 px width
    # Simple approach: keep the mask as-is; the walk will handle varying width.
    # Full skeletonisation is slow; just erode once to remove fringe pixels.
    thinned = ndimage.binary_erosion(track, iterations=1)
    if not thinned.any():
        thinned = track   # don't erase if the line is very thin already
    return _largest_component(thinned)


# ---------------------------------------------------------------------------
# Boundary / centreline ordering (connected walk)
# ---------------------------------------------------------------------------

def _walk_boundary(edge: np.ndarray) -> np.ndarray:
    """
    Walk a binary edge/line mask in traversal order using 8-connected walk.
    Returns (N, 2) array of (row, col).  Stops at gaps > 3 px.
    """
    pts = np.argwhere(edge)
    if len(pts) == 0:
        raise ValueError("Empty edge — no pixels found.")

    pt_set = {(int(r), int(c)) for r, c in pts}
    start = tuple(pts[np.lexsort((pts[:, 1], pts[:, 0]))[0]])
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
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] > 9:  # gap > ~3 px
                print(
                    f"  Warning: walk stopped early "
                    f"({len(ordered)} pts traced, {len(pt_set)} remaining). "
                    "Track line may not be fully closed.",
                    file=sys.stderr,
                )
                break
            found = tuple(remaining[nearest_idx])

        ordered.append(found)
        pt_set.discard(found)

    return np.array(ordered)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def _signed_area(pts: np.ndarray) -> float:
    """Shoelace formula — positive = CCW (image coords: col=x, row=y flipped)."""
    x, y = pts[:, 1], -pts[:, 0]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_ccw(pts: np.ndarray) -> np.ndarray:
    return pts if _signed_area(pts) >= 0 else pts[::-1]


def _align_start(reference: np.ndarray, to_align: np.ndarray) -> np.ndarray:
    dists = np.sum((to_align - reference[0]) ** 2, axis=1)
    return np.roll(to_align, -int(np.argmin(dists)), axis=0)


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed path to n evenly-spaced points by arc length."""
    closed = np.vstack([pts, pts[0]])
    seg_lens = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumlen[-1]
    targets = np.linspace(0.0, total, n, endpoint=False)
    result = np.zeros((n, 2))
    for i, t in enumerate(targets):
        idx = np.clip(int(np.searchsorted(cumlen, t, side="right")) - 1,
                      0, len(pts) - 1)
        nxt = (idx + 1) % len(pts)
        seg = seg_lens[idx]
        frac = (t - cumlen[idx]) / seg if seg > 1e-9 else 0.0
        result[i] = pts[idx] + frac * (pts[nxt] - pts[idx])
    return result


def _offset_path(centre: np.ndarray, half_width: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Offset a closed resampled centreline by ±half_width (pixels) in the
    normal direction.  Returns (left, right) in (row, col) space.
    """
    n = len(centre)
    lefts, rights = np.empty_like(centre), np.empty_like(centre)
    for i in range(n):
        prev = centre[(i - 1) % n]
        nxt  = centre[(i + 1) % n]
        tangent = nxt - prev
        norm = np.linalg.norm(tangent)
        if norm < 1e-9:
            tangent = centre[i] - prev
            norm = np.linalg.norm(tangent)
        tangent /= norm
        # Left-pointing normal in image row/col space: rotate tangent 90° CW
        # (because row increases downward)
        normal = np.array([tangent[1], -tangent[0]])
        lefts[i]  = centre[i] + half_width * normal
        rights[i] = centre[i] - half_width * normal
    return lefts, rights


def _scale_to_metres(pts: np.ndarray, img_h: int,
                     ref_pts: np.ndarray, track_length_m: float
                     ) -> np.ndarray:
    """Convert (row, col) → (x, y) in metres using ref_pts arc length."""
    def to_xy(p):
        return np.column_stack([p[:, 1], img_h - p[:, 0]])
    ref_xy = to_xy(ref_pts)
    closed = np.vstack([ref_xy, ref_xy[0]])
    pixel_len = np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1))
    scale = track_length_m / pixel_len
    return to_xy(pts) * scale


def _to_xy(pts: np.ndarray, img_h: int) -> np.ndarray:
    """Convert (row, col) → (x, y) with y-flip, in pixels."""
    return np.column_stack([pts[:, 1], img_h - pts[:, 0]])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def preview(img: np.ndarray, left: np.ndarray, right: np.ndarray,
            mode: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(img.astype(np.uint8))
    axes[0].plot(right[:, 1], right[:, 0], "y-", lw=1.5, label="Right (outer)")
    axes[0].plot(left[:, 1],  left[:, 0],  "b-", lw=1.5, label="Left (inner)")
    axes[0].scatter(right[0, 1], right[0, 0], c="lime", s=80, zorder=5)
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Detected boundaries — {mode} mode")
    axes[0].axis("off")

    axes[1].plot(right[:, 1], -right[:, 0], "y-", lw=1.5, label="Right")
    axes[1].plot(left[:, 1],  -left[:, 0],  "b-", lw=1.5, label="Left")
    axes[1].scatter(right[0, 1], -right[0, 0], c="lime", s=80, zorder=5,
                    label="Gate 0")
    axes[1].set_aspect("equal")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Extracted track (pixel coords)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Digitise a track map image into cone CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",   required=True, help="Path to the track image file")
    p.add_argument("--mode",    default="outline", choices=["outline", "ribbon"],
                   help="outline: B&W line drawing.  ribbon: filled colour band.")
    p.add_argument("--track-color", default=None, metavar="R,G,B",
                   help="Track colour as 'R,G,B' 0–255. Omit to pick interactively.")
    p.add_argument("--tolerance",   type=float, default=40.0,
                   help="Colour-match tolerance (L2, 0–255 scale)")
    p.add_argument("--half-width-px", type=float, default=8.0,
                   help="[outline] pixels to offset left/right from centreline")
    p.add_argument("--n-gates",       type=int,   default=100,
                   help="Number of gate pairs to output")
    p.add_argument("--track-length-m", type=float, default=None,
                   help="Known centreline length in metres for coordinate scaling")
    p.add_argument("--out-dir", default="data",
                   help="Output directory for left_cones.csv / right_cones.csv")
    p.add_argument("--show", action="store_true",
                   help="Preview boundaries before saving")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading image: {args.image}")
    img = load_image(args.image)
    print(f"Image size: {img.shape[1]} × {img.shape[0]} px  |  mode: {args.mode}")

    # Step 1 — identify track colour
    if args.track_color:
        parts = [float(v) for v in args.track_color.split(",")]
        if len(parts) != 3:
            raise ValueError("--track-color must be 'R,G,B' e.g. '0,0,0'")
        track_rgb = np.array(parts, dtype=float)
    else:
        track_rgb = pick_color_interactively(img)

    # Step 2 — build mask
    print(f"Building track mask (tolerance={args.tolerance}) …")
    if args.mode == "outline":
        mask = build_mask_outline(img, track_rgb, args.tolerance)
    else:
        mask = build_mask_ribbon(img, track_rgb, args.tolerance)

    frac = mask.sum() / mask.size
    print(f"  Track pixels: {mask.sum():,} ({frac:.1%} of image)")
    if frac < 0.002:
        print("  Warning: very few pixels matched — try increasing --tolerance "
              "or double-check --track-color.", file=sys.stderr)
    if frac > 0.5:
        print("  Warning: >50% of image matched — try decreasing --tolerance "
              "or invert your colour choice (use background colour instead).",
              file=sys.stderr)

    # Step 3 — extract path(s)
    if args.mode == "outline":
        print("Extracting centreline …")
        centre_mask = extract_centreline_mask(mask)
        print(f"  Centreline pixels: {centre_mask.sum():,}")
        centre_raw  = _walk_boundary(centre_mask)
        centre_ccw  = _ensure_ccw(centre_raw)
        print(f"Resampling to {args.n_gates} points …")
        centre_rs   = _resample(centre_ccw, args.n_gates)
        left_rc, right_rc = _offset_path(centre_rs, args.half_width_px)

    else:  # ribbon
        print("Extracting ribbon edges …")
        outer_edge, inner_edge = extract_ribbon_edges(mask)
        print(f"  Outer: {outer_edge.sum():,} px  |  Inner: {inner_edge.sum():,} px")
        outer_raw = _walk_boundary(outer_edge)
        inner_raw = _walk_boundary(inner_edge)
        outer_ccw = _ensure_ccw(outer_raw)
        inner_ccw = _ensure_ccw(inner_raw)
        inner_ccw = _align_start(outer_ccw, inner_ccw)
        print(f"Resampling to {args.n_gates} gates …")
        right_rc = _resample(outer_ccw, args.n_gates)   # outer = right
        left_rc  = _resample(inner_ccw, args.n_gates)   # inner = left

    # Step 4 — optional preview (in row/col space)
    if args.show:
        preview(img, left_rc, right_rc, args.mode)

    # Step 5 — convert to output coordinates
    ref = right_rc  # use right/outer boundary as length reference
    if args.track_length_m:
        left_out  = _scale_to_metres(left_rc,  img.shape[0], ref, args.track_length_m)
        right_out = _scale_to_metres(right_rc, img.shape[0], ref, args.track_length_m)
        unit = "m"
    else:
        left_out  = _to_xy(left_rc,  img.shape[0])
        right_out = _to_xy(right_rc, img.shape[0])
        unit = "px"

    # Step 6 — save
    out_dir    = Path(args.out_dir)
    left_path  = out_dir / "left_cones.csv"
    right_path = out_dir / "right_cones.csv"
    pd.DataFrame(left_out,  columns=["x", "y"]).round(4).to_csv(left_path,  index=False)
    pd.DataFrame(right_out, columns=["x", "y"]).round(4).to_csv(right_path, index=False)
    print(f"Saved {args.n_gates} gate pairs ({unit}) →")
    print(f"  {left_path}  (left cones)")
    print(f"  {right_path} (right cones)")


if __name__ == "__main__":
    main()
