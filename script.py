"""
Track Path Optimiser — entry point.

Orchestrates the pipeline: load → centerline → optimise → smooth → speed profile → visualise.
"""

import argparse
from track import load_track, resample_track, gate_midpoints, gate_widths, is_loop_closed
from optimiser import optimise
from visualiser import plot_track


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FSAE Track Path Optimiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--track",     default="data/track.csv", help="Track CSV path")

    # Vehicle constraints
    p.add_argument("--a-lat",     type=float, default=12.0,  help="Max lateral acceleration (m/s²)")
    p.add_argument("--a-lon",     type=float, default=8.0,   help="Max longitudinal acceleration / braking (m/s²)")
    p.add_argument("--v-max",     type=float, default=15.0,  help="Max speed (m/s)")
    p.add_argument("--v-min",     type=float, default=1.0,   help="Min speed on velocity grid (m/s)")
    p.add_argument("--mass",      type=float, default=230.0, help="Vehicle mass (kg)")
    p.add_argument("--c-drag",    type=float, default=0.5,   help="Drag coefficient × frontal area (kg/m)")
    p.add_argument("--wheelbase", type=float, default=1.55,  help="Vehicle wheelbase (m)")

    # Gate count
    p.add_argument("--resample-gates", type=int, default=None,
                   help="Resample track to this many gates before running "
                        "(e.g. 200). Uses cubic spline on both boundaries.")

    # Optimiser resolution
    p.add_argument("--n-samples", type=int,   default=11,    help="Lateral position samples per gate")
    p.add_argument("--n-vel",     type=int,   default=11,    help="Velocity grid resolution")

    return p


def main() -> None:
    args = build_parser().parse_args()

    # --- Stage 1 & 2: Load, optionally resample, compute centerline ---
    left_cones, right_cones = load_track(args.track)
    if args.resample_gates:
        left_cones, right_cones = resample_track(left_cones, right_cones,
                                                  args.resample_gates)
        print(f"Resampled: {args.resample_gates} gates")

    closed     = is_loop_closed(left_cones, right_cones)
    centerline = gate_midpoints(left_cones, right_cones)
    widths     = gate_widths(left_cones, right_cones)

    print(f"Loaded   : {len(left_cones)} gates from '{args.track}'")
    print(f"Closed   : {closed}")
    print(f"Width    : min={widths.min():.2f}  mean={widths.mean():.2f}  max={widths.max():.2f} m")

    # --- Stage 3: Physics DP optimiser ---
    print(f"Running DP  N_lat={args.n_samples}  N_vel={args.n_vel}  "
          f"v=[{args.v_min}, {args.v_max}] m/s  "
          f"a_lat={args.a_lat}  a_lon={args.a_lon} m/s²  "
          f"mass={args.mass} kg  c_drag={args.c_drag} kg/m …")

    dp_path, dp_speeds = optimise(
        left_cones, right_cones,
        n_lat=args.n_samples,
        n_vel=args.n_vel,
        v_max=args.v_max,
        v_min=args.v_min,
        a_lat_max=args.a_lat,
        a_lon_max=args.a_lon,
        mass=args.mass,
        c_drag=args.c_drag,
    )
    print(f"DP done  : {len(dp_path)} waypoints  "
          f"speed min={dp_speeds.min():.1f}  mean={dp_speeds.mean():.1f}  max={dp_speeds.max():.1f} m/s")

    plot_track(
        left_cones, right_cones,
        centerline=centerline,
        dp_path=dp_path,
        speed_profile=dp_speeds,
        closed=closed,
        title=f"Track Map — DP Path  (N_lat={args.n_samples}, N_vel={args.n_vel})",
    )


if __name__ == "__main__":
    main()
