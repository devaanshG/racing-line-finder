"""
Track Path Optimiser — entry point.

Orchestrates the pipeline: load → centerline → optimise → smooth → speed profile → visualise.
"""

import argparse
from track import load_track
from visualiser import plot_track


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FSAE Track Path Optimiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--track",    default="data/track.csv",  help="Track CSV path")

    # Vehicle constraints (used from Stage 5 onward)
    p.add_argument("--a-lat",     type=float, default=12.0,  help="Max lateral acceleration (m/s²)")
    p.add_argument("--v-max",     type=float, default=15.0,  help="Max speed (m/s)")
    p.add_argument("--wheelbase", type=float, default=1.55,  help="Vehicle wheelbase (m)")

    # Optimiser settings (used from Stage 3 onward)
    p.add_argument("--n-samples", type=int,   default=11,    help="DP gate resolution (samples per gate)")
    p.add_argument("--w-len",     type=float, default=1.0,   help="DP cost weight: path length")
    p.add_argument("--w-curve",   type=float, default=0.5,   help="DP cost weight: curvature penalty")

    return p


def main() -> None:
    args = build_parser().parse_args()

    # --- Stage 1: Load & visualise ---
    left_cones, right_cones = load_track(args.track)
    print(f"Loaded {len(left_cones)} gate pairs from '{args.track}'.")
    plot_track(left_cones, right_cones)


if __name__ == "__main__":
    main()
