# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FSAE Track Path Optimiser — a Python program that computes a fast, smooth driving path through a Formula Student track defined by left and right cone boundaries.

**Final output:** an array of `(x, y, v)` waypoints representing the racing line and target speed at each point.

## Architecture

```
script.py           ← CLI entry point; orchestrates the pipeline
track.py            ← cone loading, gate generation, centerline
optimiser.py        ← dynamic programming graph build + solve
smoother.py         ← spline fitting, boundary violation check
speed_profile.py    ← forward-backward velocity pass
visualiser.py       ← all matplotlib plotting
data/
  track.csv         ← track definition: left_x, left_y, right_x, right_y per gate
```

## Input Format

A single CSV with one row per gate. Each row defines the left and right cone positions for that gate.

```
left_x,left_y,right_x,right_y
1.0,0.0,4.0,0.0
2.5,3.2,5.5,3.0
...
```

Vehicle constraints are passed as CLI arguments (see Running section).

## Development Stages (Waterfall)

Each stage produces a working, runnable deliverable.

| Stage | Goal | Key Output |
|-------|------|------------|
| 1 | Load cones & render track | Static track map plot |
| 2 | Gate generation & centerline | Centerline overlaid on track |
| 3 | DP path optimiser | Raw optimised path through gates |
| 4 | Path smoothing | Smooth spline path, bound-checked |
| 5 | Speed profile | `(x, y, v)` output + speed plot |

## Design Decisions

- **Cone pairing:** Single CSV with left/right columns per row. No matching algorithm needed.
- **Curvature (DP stage):** 3-point finite difference — simple and sufficient for cost scoring.
- **Smoothing:** Parametric cubic spline (`scipy.interpolate.splprep`) — handles closed loops.
- **Speed profile:** Forward-backward pass. `v_max(s) = sqrt(a_lat / |κ|)`, clamped to `v_max`.
- **Gate resolution N:** Configurable (default 11). Controls DP quality vs. runtime trade-off.

## Code Standards

- Each module has a single, clear responsibility. No cross-module side effects.
- All configurable parameters (N, cost weights, vehicle constraints) are either CLI args or named constants at the top of the relevant module — never magic numbers buried in logic.
- Functions are short and named for what they return, not what they do (`gate_midpoints()` not `compute_midpoints()`).
- No external dependencies beyond: `numpy`, `scipy`, `matplotlib`, `pandas`. Do not introduce others without noting it in CHANGELOG.md.
- All plots are produced by `visualiser.py` only. No `plt` calls in other modules.
- Verify boundary containment after smoothing. If violations exist, warn clearly — do not silently discard them.

## Running

```bash
# Minimal — uses defaults
python script.py --track data/track.csv

# Full options
python script.py \
  --track data/track.csv \
  --a-lat 12.0 \        # max lateral acceleration (m/s²)
  --v-max 15.0 \        # max speed (m/s)
  --wheelbase 1.55 \    # vehicle wheelbase (m)
  --n-samples 11 \      # DP gate resolution
  --w-len 1.0 \         # DP cost weight: path length
  --w-curve 0.5         # DP cost weight: curvature penalty
```

## Changelog

All decisions, stage completions, and dependency additions are recorded in [CHANGELOG.md](CHANGELOG.md).
