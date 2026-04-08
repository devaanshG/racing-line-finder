"""
Track Path Optimiser — entry point.

Orchestrates the pipeline: load → centerline → optimise → smooth → speed profile → visualise.
"""

import argparse
from track import load_track, gate_midpoints, gate_widths, is_loop_closed
from optimiser import optimise
from visualiser import plot_track


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FSAE Track Path Optimiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--track",     default="data/track.csv", help="Track CSV path")

    # Vehicle constraints (used from Stage 5 onward)
    p.add_argument("--a-lat",     type=float, default=12.0,  help="Max lateral acceleration (m/s²)")
    p.add_argument("--v-max",     type=float, default=15.0,  help="Max speed (m/s)")
    p.add_argument("--wheelbase", type=float, default=1.55,  help="Vehicle wheelbase (m)")

    # Optimiser settings (used from Stage 3 onward)
    p.add_argument("--n-samples", type=int,   default=11,    help="DP gate resolution (samples per gate)")
    p.add_argument("--w-len",     type=float, default=1.0,   help="DP cost weight: path length")
    p.add_argument("--w-curve",   type=float, default=5.0,   help="DP cost weight: heading-change² penalty (increase for wider arcs)")

    return p


def main() -> None:
    args = build_parser().parse_args()

    # --- Stage 1 & 2: Load, compute centerline, visualise ---
    left_cones, right_cones = load_track(args.track)

    closed = is_loop_closed(left_cones, right_cones)
    centerline = gate_midpoints(left_cones, right_cones)
    widths = gate_widths(left_cones, right_cones)

    print(f"Loaded   : {len(left_cones)} gates from '{args.track}'")
    print(f"Closed   : {closed}")
    print(f"Width    : min={widths.min():.2f}  mean={widths.mean():.2f}  max={widths.max():.2f}")

    # --- Stage 3: DP optimiser ---
    print(f"Running DP (N={args.n_samples}, w_len={args.w_len}, w_curve={args.w_curve}) …")
    dp_path = optimise(
        left_cones, right_cones,
        n_samples=args.n_samples,
        w_len=args.w_len,
        w_curve=args.w_curve,
    )
    print(f"DP done  : path has {len(dp_path)} waypoints")

    plot_track(
        left_cones, right_cones,
        centerline=centerline,
        dp_path=dp_path,
        closed=closed,
        title=f"Track Map — DP Path (N={args.n_samples}, "
              f"w_len={args.w_len}, w_curve={args.w_curve})",
    )


if __name__ == "__main__":
    main()
