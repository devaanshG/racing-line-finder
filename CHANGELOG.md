# CHANGELOG

Records all decisions, stage completions, and notable changes in the Track Path Optimiser project.
Format: `[YYYY-MM-DD] Type — Description`

Types: `DECISION`, `STAGE`, `FIX`, `DEPENDENCY`, `ASSUMPTION`

---

## Stage 2 — Gate Generation & Centerline (2026-04-08)

### STAGE — Stage 2 complete
Deliverable: `python script.py` now prints gate stats and plots the naive gate-midpoint centerline overlaid on the track boundaries.
Files changed: `track.py` (new helpers), `visualiser.py` (centerline overlay), `script.py` (wires them together).

### DECISION — Centerline = gate midpoints at Stage 2
The naive centerline is simply `(left + right) / 2` at each gate. No smoothing or optimisation at this stage.
**Rationale:** Provides a baseline path that is guaranteed to stay in bounds; the DP optimiser (Stage 3) will improve on it.

### DECISION — Loop closure detected automatically
`is_loop_closed()` compares the distance between first and last gate midpoints against 5× the mean gate spacing.
**Rationale:** Avoids hardcoding the assumption; handles both open (autocross slalom) and closed (circuit) tracks without a CLI flag.

### DECISION — `plot_track` extended with optional `centerline` parameter
Rather than a separate `plot_centerline` function, the centerline is an optional overlay on the existing track plot.
**Rationale:** Stage 1 output (no centerline) still works with the same call; Stage 2 just passes the extra array. Avoids duplicating the figure setup.

---

## 2026-04-08

### DECISION — Cone pairing by row index
Input CSVs are pre-paired: row `i` in `left_cones.csv` and row `i` in `right_cones.csv` always form a gate.
No nearest-neighbour matching algorithm is needed. This simplifies gate generation significantly.
**Source:** User clarification.

### DECISION — Waterfall development with per-stage deliverables
Stages: (1) Loader + Visualiser → (2) Gates + Centerline → (3) DP Optimiser → (4) Smoother → (5) Speed Profile.
Each stage leaves the program in a runnable state before the next begins.
**Rationale:** Allows early validation of track geometry before investing in optimisation logic.

### DECISION — Dynamic programming over continuous optimisation
Used DP on a discretised gate graph rather than gradient-based optimisation (e.g. scipy.minimize on spline control points).
**Rationale:** DP is explainable, easy to debug visually, and the approach is well-validated in FSAE path planning literature. Continuous optimisation is listed as a future extension.

### DECISION — 3-point finite difference for curvature in DP
Curvature at node B (between A and B and C) estimated as the inverse radius of the circumscribed circle through A, B, C.
**Rationale:** No spline needed at the DP stage; keeps the graph-build fast and avoids circular dependency with the smoother.

### DECISION — Parametric cubic spline for smoothing
`scipy.interpolate.splprep` used for smoothing. Handles closed-loop tracks naturally via `per=True`.
**Rationale:** Built-in to scipy, well-tested, produces C2-continuous curves.

### DECISION — Forward-backward pass for speed profile
Standard racing-line velocity planning: curvature-limited `v_max`, forward pass (acceleration), backward pass (braking).
**Rationale:** O(n), physically interpretable, easy to tune with `a_lon` parameter.

### ASSUMPTION — Track is a closed loop
The last gate connects back to the first. The start position lies near gate 0.
**Implication:** Spline smoothing uses `per=True`; DP graph wraps last gate back to first.

### ASSUMPTION — Cones are ordered along the track direction ✓ confirmed
Both CSVs list cones in traversal order (gate 0 is the start, gate N-1 is just before the finish line).
**Implication:** No re-ordering or TSP-style sorting is applied to the input.

### DEPENDENCY — Initial allowed dependencies
`numpy`, `scipy`, `matplotlib`, `pandas`. Any additions must be logged here.

---

## Stage 1 — Track Loader & Visualiser (2026-04-08)

### STAGE — Stage 1 complete
Deliverable: `python script.py` loads cone CSVs and renders a static track map.
Files: `script.py` (CLI), `track.py` (load + validate), `visualiser.py` (plot), `data/generate_sample.py` (sample data generator).

### DECISION — All CLI args declared at Stage 1
Vehicle constraints (`--a-lat`, `--v-max`, `--wheelbase`) and optimiser settings (`--n-samples`, `--w-len`, `--w-curve`) are defined in the parser now, even though they are unused until later stages.
**Rationale:** Avoids breaking the CLI interface mid-project; args are simply ignored until their stage is reached.

### DECISION — `gate_midpoints()` added to track.py at Stage 1
The naive centerline is computed in `track.py` as `(left + right) / 2`. It is not used by the visualiser in Stage 1 but is ready for Stage 2.
**Rationale:** Keeps gate logic co-located and avoids a redundant pass over the arrays later.

### DECISION — Track digitiser from image added (data/from_image.py)
Supports loading any top-down track map image (PNG, JPG, etc.) and converting it to left/right cone CSVs.
Two modes: interactive colour-pick (click on the track) or `--track-color R,G,B` CLI argument.
Uses scipy.ndimage morphological operations to extract inner/outer boundaries; connected walk to order pixels.
Outer boundary → right_cones.csv; inner boundary → left_cones.csv.
Optional `--track-length-m` scales pixel coords to metres.
**New dependency:** Pillow (added to requirements.txt).

### DECISION — Sample track: oval with east/west hairpins
A symmetric oval (~88 gates, 36 m × 16 m centreline, 3 m track width) generated by `data/generate_sample.py`.
Centreline offsets use averaged tangent normals to handle corners without self-intersection.
**Rationale:** Simple enough to verify visually; complex enough to exercise both straights and curves in later stages.
