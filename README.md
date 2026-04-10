# FSAE Track Path Optimiser

Computes the minimum-time racing line through a Formula Student track defined by left/right cone boundaries. Outputs an array of `(x, y, v)` waypoints — position and speed at each gate.

## How it works

The track is divided into **gates** (pairs of left/right cones). At each gate the car can be at one of N lateral positions and one of N speeds, forming a 3D graph of nodes `(gate, lateral_position, velocity)`. A **dynamic programming** algorithm finds the path through this graph that minimises total lap time.

The racing line emerges from physics: a wider arc through a corner has a larger turning radius, which allows a higher corner speed and therefore less time. The algorithm finds this automatically by computing the exact circumradius of every three-point arc `A → B → C` and penalising transitions that would exceed tyre grip.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib pandas
```

## Quick start

```bash
# Run on the included sample oval
python script.py --track data/track.csv
```

## Full CLI reference

```
python script.py [options]
```

### Track input

| Argument | Default | Description |
|---|---|---|
| `--track` | `data/track.csv` | Path to track CSV |
| `--resample-gates N` | off | Resample track to N evenly-spaced gates before running (cubic spline on both boundaries). Useful for coarse hand-digitised tracks. |

### Vehicle constraints

| Argument | Default | Description |
|---|---|---|
| `--a-lat` | `12.0` | Max lateral (cornering) acceleration m/s² |
| `--a-lon` | `8.0` | Max longitudinal acceleration / braking m/s² |
| `--v-max` | `15.0` | Maximum speed m/s |
| `--v-min` | `1.0` | Minimum speed on the velocity grid m/s |
| `--mass` | `230.0` | Vehicle mass kg (driver included) |
| `--c-drag` | `0.5` | Aerodynamic drag coefficient × frontal area kg/m |
| `--wheelbase` | `1.55` | Vehicle wheelbase m |

### Optimiser resolution

| Argument | Default | Description |
|---|---|---|
| `--n-samples` | `11` | Lateral position samples per gate. More = finer racing line, slower. |
| `--n-vel` | `11` | Velocity grid resolution. More = finer speed profile, slower. |

### Example — custom vehicle

```bash
python script.py \
  --track data/track.csv \
  --a-lat 14.0 \
  --a-lon 10.0 \
  --v-max 18.0 \
  --mass 210.0 \
  --c-drag 0.4 \
  --n-samples 15 \
  --n-vel 15
```

### Example — resampling a coarse digitised track

```bash
python script.py --track data/track.csv --resample-gates 200
```

## Track CSV format

One row per gate, in driving order. Each row gives the left and right cone positions for that gate.

```csv
left_x,left_y,right_x,right_y
1.0,0.0,4.0,0.0
2.5,3.2,5.5,3.0
...
```

Units should be consistent (metres recommended). Coordinates can be in any reference frame.

## Creating a track

### From an image (recommended)

Digitise any top-down track map photo:

```bash
python data/from_image.py --image track.png
```

Click the centreline in driving order. Right-click to undo. Press **Enter** when done. Gates are generated automatically by offsetting the fitted spline by `--half-width-px` pixels on each side.

```
Options:
  --image           Path to track image (PNG, JPG, …)
  --mode            manual | outline | ribbon  (default: manual)
  --half-width-px   Pixels from centreline to each cone row (default: 8)
  --n-gates         Number of gate pairs to output (default: 100)
  --track-length-m  Known real-world length for coordinate scaling
  --no-loop         Treat track as open (not closed)
  --show            Preview gates before saving
```

### Generate the sample oval

```bash
python data/generate_sample.py            # 88 gates (default)
python data/generate_sample.py --n-gates 200
```

Produces a closed oval (~36 m × 16 m, 3 m wide) with two hairpins and two straights.

## Output

The visualiser shows the track on a black background with:
- White boundary lines
- Dashed orange centreline
- Racing line coloured by speed (plasma colourmap — dark purple = slow, bright yellow = fast)
- Green star = start point and heading direction

## Project structure

```
script.py           — entry point, CLI, pipeline orchestration
track.py            — CSV loading, gate midpoints, resampling
optimiser.py        — physics DP: graph build and solve
visualiser.py       — all matplotlib plotting
data/
  track.csv         — track definition (left_x, left_y, right_x, right_y)
  from_image.py     — interactive track digitiser from top-down image
  generate_sample.py — oval sample track generator
CHANGELOG.md        — decision log
```

## Algorithm complexity

With `N` lateral samples, `V` velocity steps, and `G` gates:

| Parameter | Formula | At N=V=11, G=100 |
|---|---|---|
| Nodes per gate | N × V × N | 1,331 |
| Operations per gate | N³ × V² | ~161,000 |
| Total operations | G × N³ × V² | ~16 M |
| Typical runtime | — | < 0.1 s |

## Dependencies

`numpy` `scipy` `matplotlib` `pandas`

No other dependencies. Python 3.10+.
