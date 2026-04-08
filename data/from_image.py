"""
Digitise a top-down track map image into left_cones.csv / right_cones.csv.

Usage
-----
Interactive (click on track surface to identify colour):
    python data/from_image.py --image path/to/track.png

With explicit track colour (skip the click step):
    python data/from_image.py --image path/to/track.png --track-color 220,50,50

With real-world scaling (provide the known track length):
    python data/from_image.py --image path/to/track.png --track-length-m 5300

Options
-------
--image          Path to the track image (PNG, JPG, …)
--track-color    Track surface colour as R,G,B integers (0–255). If omitted,
                 a window opens and you click once on the track surface.
--tolerance      Colour-match tolerance in [0–255] space (default: 40)
--n-gates        Number of gate pairs to output (default: 100)
--track-length-m Known total outer-boundary length in metres for scaling.
                 If omitted, coordinates are in pixels.
--out-dir        Where to write the CSVs (default: data/)
--show           Display the extracted boundaries before saving.

Algorithm
---------
1. Load image → float RGB array.
2. Build binary track mask via colour-distance threshold.
3. Morphological cleanup (remove noise, fill small gaps).
4. Separate outer edge (track ∩ background-adjacent) and inner edge
   (track ∩ infield-adjacent) using scipy.ndimage operations.
5. Walk each 1-px boundary with a connected-component traversal → ordered path.
6. Align inner path to outer path (same start angle).
7. Ensure both paths are counter-clockwise; reverse if needed.
8. Resample each path to --n-gates evenly-spaced points.
9. Optionally scale pixels → metres.
10. Save CSVs; outer → right_cones, inner → left_cones.
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
    """Load image as float32 RGB array with values in [0, 255]."""
    img = mpimg.imread(path)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.float32)
    else:
        img = img.astype(np.float32)
    if img.ndim == 2:                    # grayscale → RGB
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:               # RGBA → RGB
        img = img[:, :, :3]
    return img


# ---------------------------------------------------------------------------
# Colour picking
# ---------------------------------------------------------------------------

def pick_color_interactively(img: np.ndarray) -> np.ndarray:
    """
    Open the image in a matplotlib window. The user clicks once on the track
    surface; the sampled RGB is returned.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img.astype(np.uint8))
    ax.set_title("Click ONCE on the track surface, then close this window", fontsize=12)
    ax.axis("off")
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not pts:
        raise RuntimeError("No point selected — aborting.")
    x, y = int(round(pts[0][0])), int(round(pts[0][1]))
    colour = img[y, x, :3]
    print(f"Sampled track colour at ({x}, {y}): RGB = {colour.astype(int)}")
    return colour


def build_track_mask(img: np.ndarray, track_rgb: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Binary mask: True where the pixel colour is within `tolerance` (L2 in RGB
    space) of `track_rgb`.  Returns a bool array of shape (H, W).
    """
    diff = img[:, :, :3].astype(float) - track_rgb.astype(float)
    dist = np.linalg.norm(diff, axis=-1)
    mask = dist < tolerance

    # Morphological cleanup: remove speckle, bridge tiny gaps
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=3)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Boundary extraction
# ---------------------------------------------------------------------------

def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return a mask containing only the largest connected component."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        raise ValueError("No connected component found in mask.")
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return (labeled == (np.argmax(sizes) + 1)).astype(bool)


def extract_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    From a binary track-ring mask, return:
        outer_edge — track pixels adjacent to the outside (background)
        inner_edge — track pixels adjacent to the infield

    Both are bool arrays of shape (H, W).

    Raises ValueError if no infield can be detected (e.g. a straight line
    rather than a closed ring).
    """
    track = _largest_component(mask)
    filled = ndimage.binary_fill_holes(track)

    # Infield = the hole(s) inside the ring
    infield_all = filled & ~track
    if not infield_all.any():
        raise ValueError(
            "Could not detect an infield. "
            "The track mask must form a closed ring with a hole inside."
        )
    infield = _largest_component(infield_all)

    # Background = everything outside the filled track
    background = ~filled

    outer_edge = track & ndimage.binary_dilation(background)
    inner_edge = track & ndimage.binary_dilation(infield)
    return outer_edge, inner_edge


# ---------------------------------------------------------------------------
# Contour ordering (connected walk)
# ---------------------------------------------------------------------------

def _walk_boundary(edge: np.ndarray) -> np.ndarray:
    """
    Walk a binary edge mask (should be ~1 px thick) in order using a
    connected-component traversal.  Returns an (N, 2) array of (row, col).

    The walk starts at the topmost-leftmost pixel and proceeds using
    8-connectivity, always choosing the nearest unvisited neighbour.
    Any gap larger than √2 pixels (non-adjacent) ends the walk.
    """
    pts = np.argwhere(edge)
    if len(pts) == 0:
        raise ValueError("Empty edge — no boundary pixels found.")

    # Build a set for O(1) membership tests
    pt_set = {(r, c) for r, c in pts}

    # Start from topmost-leftmost
    start = tuple(pts[np.lexsort((pts[:, 1], pts[:, 0]))[0]])
    ordered = [start]
    pt_set.discard(start)

    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    while pt_set:
        r, c = ordered[-1]
        # Check 8-connected neighbours first (fast path)
        found = None
        for dr, dc in offsets:
            nb = (r + dr, c + dc)
            if nb in pt_set:
                found = nb
                break

        if found is None:
            # Nearest unvisited point (handles tiny gaps from morphological ops)
            remaining = np.array(list(pt_set))
            dists = np.sum((remaining - np.array([r, c])) ** 2, axis=1)
            nearest_idx = np.argmin(dists)
            if dists[nearest_idx] > 10:   # gap > ~3 px → stop
                print(
                    f"  Warning: boundary walk stopped early "
                    f"({len(ordered)} pts traced, {len(pt_set)} remaining).",
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
    """Shoelace formula — positive = CCW, negative = CW."""
    x, y = pts[:, 1], pts[:, 0]   # col → x, row → y (flipped)
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_ccw(pts: np.ndarray) -> np.ndarray:
    """Reverse the path if it is clockwise."""
    return pts if _signed_area(pts) >= 0 else pts[::-1]


def _align_start(reference: np.ndarray, to_align: np.ndarray) -> np.ndarray:
    """
    Rotate `to_align` so that its starting point is the one closest to
    `reference[0]`.
    """
    dists = np.sum((to_align - reference[0]) ** 2, axis=1)
    offset = int(np.argmin(dists))
    return np.roll(to_align, -offset, axis=0)


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed path to exactly n evenly-spaced points by arc length."""
    # Close the loop for length calculation
    closed = np.vstack([pts, pts[0]])
    seg_lens = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumlen[-1]

    targets = np.linspace(0.0, total, n, endpoint=False)
    result = np.zeros((n, 2))
    for i, t in enumerate(targets):
        idx = int(np.searchsorted(cumlen, t, side="right")) - 1
        idx = np.clip(idx, 0, len(pts) - 1)
        next_idx = (idx + 1) % len(pts)
        seg = seg_lens[idx]
        frac = (t - cumlen[idx]) / seg if seg > 1e-9 else 0.0
        result[i] = pts[idx] + frac * (pts[next_idx] - pts[idx])
    return result


def _pixels_to_metres(outer_px: np.ndarray, inner_px: np.ndarray,
                       img_h: int, track_length_m: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel (row, col) → metric (x, y) coordinates.
    Scaling is derived from the outer boundary arc length matched to
    `track_length_m`.
    """
    def to_xy(pts):
        return np.column_stack([pts[:, 1], img_h - pts[:, 0]])

    outer_xy = to_xy(outer_px)
    inner_xy = to_xy(inner_px)

    closed = np.vstack([outer_xy, outer_xy[0]])
    pixel_len = np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1))
    scale = track_length_m / pixel_len
    return outer_xy * scale, inner_xy * scale


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def preview(img: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: raw image with overlaid boundaries
    axes[0].imshow(img.astype(np.uint8))
    axes[0].plot(outer[:, 1], outer[:, 0], "y-", linewidth=1.5, label="Outer (right)")
    axes[0].plot(inner[:, 1], inner[:, 0], "b-", linewidth=1.5, label="Inner (left)")
    axes[0].scatter(outer[0, 1], outer[0, 0], c="lime", s=80, zorder=5, label="Gate 0")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Detected boundaries")
    axes[0].axis("off")

    # Right: extracted path in coordinate space
    axes[1].plot(outer[:, 1], -outer[:, 0], "y-", linewidth=1.5, label="Outer (right)")
    axes[1].plot(inner[:, 1], -inner[:, 0], "b-", linewidth=1.5, label="Inner (left)")
    axes[1].scatter(outer[0, 1], -outer[0, 0], c="lime", s=80, zorder=5, label="Gate 0")
    axes[1].set_aspect("equal")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Extracted track (pixel coords)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Digitise a track map image into cone CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", required=True, help="Path to the track image file")
    p.add_argument(
        "--track-color", default=None, metavar="R,G,B",
        help="Track surface colour as 'R,G,B' integers 0–255. "
             "Omit to pick interactively.",
    )
    p.add_argument("--tolerance", type=float, default=40.0,
                   help="Colour-match tolerance (L2 in RGB space, 0–255 scale)")
    p.add_argument("--n-gates", type=int, default=100,
                   help="Number of gate pairs to output")
    p.add_argument("--track-length-m", type=float, default=None,
                   help="Known outer-boundary length in metres for scaling. "
                        "If omitted, output is in pixels.")
    p.add_argument("--out-dir", default="data",
                   help="Directory to write left_cones.csv / right_cones.csv")
    p.add_argument("--show", action="store_true",
                   help="Preview detected boundaries before saving")
    return p


def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading image: {args.image}")
    img = load_image(args.image)
    print(f"Image size: {img.shape[1]} × {img.shape[0]} px")

    # --- Step 1: identify track colour ---
    if args.track_color:
        parts = [float(v) for v in args.track_color.split(",")]
        if len(parts) != 3:
            raise ValueError("--track-color must be 'R,G,B' e.g. '220,50,50'")
        track_rgb = np.array(parts, dtype=float)
    else:
        track_rgb = pick_color_interactively(img)

    # --- Step 2: build mask ---
    print(f"Building track mask (tolerance={args.tolerance}) …")
    mask = build_track_mask(img, track_rgb, args.tolerance)
    frac = mask.sum() / mask.size
    print(f"  Track pixels: {mask.sum():,} ({frac:.1%} of image)")
    if frac < 0.005:
        print("  Warning: very few track pixels detected — try increasing --tolerance",
              file=sys.stderr)

    # --- Step 3: extract edges ---
    print("Extracting boundaries …")
    outer_edge, inner_edge = extract_edges(mask)
    print(f"  Outer edge pixels: {outer_edge.sum():,}")
    print(f"  Inner edge pixels: {inner_edge.sum():,}")

    # --- Step 4: walk boundaries into ordered paths ---
    print("Ordering boundaries …")
    outer_raw = _walk_boundary(outer_edge)
    inner_raw = _walk_boundary(inner_edge)
    print(f"  Outer path: {len(outer_raw)} pts, Inner path: {len(inner_raw)} pts")

    # --- Step 5: normalise direction and alignment ---
    outer_ccw = _ensure_ccw(outer_raw)
    inner_ccw = _ensure_ccw(inner_raw)
    inner_aligned = _align_start(outer_ccw, inner_ccw)

    # --- Step 6: resample ---
    print(f"Resampling to {args.n_gates} gates …")
    outer_rs = _resample(outer_ccw, args.n_gates)
    inner_rs = _resample(inner_aligned, args.n_gates)

    # --- Step 7: optional preview ---
    if args.show:
        preview(img, outer_rs, inner_rs)

    # --- Step 8: optionally scale to metres ---
    if args.track_length_m:
        outer_out, inner_out = _pixels_to_metres(
            outer_rs, inner_rs, img.shape[0], args.track_length_m
        )
        unit = "m"
    else:
        # Convert (row, col) → (x, y) with y-flip
        def to_xy(pts):
            return np.column_stack([pts[:, 1], img.shape[0] - pts[:, 0]])
        outer_out = to_xy(outer_rs)
        inner_out = to_xy(inner_rs)
        unit = "px"

    # --- Step 9: save ---
    out_dir = Path(args.out_dir)
    left_path  = out_dir / "left_cones.csv"
    right_path = out_dir / "right_cones.csv"
    pd.DataFrame(inner_out, columns=["x", "y"]).round(4).to_csv(left_path,  index=False)
    pd.DataFrame(outer_out, columns=["x", "y"]).round(4).to_csv(right_path, index=False)
    print(f"Saved {args.n_gates} gate pairs ({unit}) to:")
    print(f"  {left_path}  (inner / left cones)")
    print(f"  {right_path} (outer / right cones)")


if __name__ == "__main__":
    main()
