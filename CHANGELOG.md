# CHANGELOG

Records all decisions, stage completions, and notable changes in the Track Path Optimiser project.
Format: `[YYYY-MM-DD] Type — Description`

Types: `DECISION`, `STAGE`, `FIX`, `DEPENDENCY`, `ASSUMPTION`

---

## Stage 3 — Physics DP upgrade (2026-04-09)

### STAGE — Stage 3 revised: geometric DP → physics DP
State space changed from `(gate, lat, prev_lat)` to `(gate, lat, velocity)`.
The objective is now minimum lap time in seconds, not a weighted proxy.
Output extended from `path (G,2)` to `(path (G,2), speeds (G,))` — jointly optimal.
Files changed: `optimiser.py` (full rewrite), `script.py` (new args, unpack tuple), `visualiser.py` (speed-coloured LineCollection overlay).

### DECISION — Replace geometric cost with physics-based lap-time cost
The previous `w_len·dist + w_curve·θ²` cost was dimensionless and required non-physical weight tuning. The physics DP minimises `Σ dist/v_avg` (total time in seconds) with hard penalty walls for infeasible transitions. All parameters now have physical meaning and can be measured from the car.

### DECISION — State: (lat_j, vel_k) replaces (lat_j, prev_lat_k)
`prev_lat` was needed to compute θ² at the current node. In the physics DP `velocity` serves the same role as the "memory" variable — needed to compute transition time and force demands — so no look-ahead is required. The state size is identical: N_lat × N_vel = 11 × 11 = 121 nodes per gate.

### DECISION — Penalty values set to 1e6 seconds
Following the referenced approach, infeasible transitions are penalised with 1e6 s (≈11.6 days). This makes them effectively unreachable without requiring hard pruning.
Three penalties: PENALTY_DIRECTION (displacement opposes gate flow), PENALTY_FORCE for lateral accel > a_lat_max, and PENALTY_FORCE for longitudinal demand > a_lon_max.

### DECISION — Curvature approximation: |sin α| / dist
κ is approximated from the angle between the displacement vector B→C and the track forward direction. |sin α| is the cross-product magnitude, giving the lateral component per unit distance. Simpler than the 3-point circumradius and does not require the previous gate position.

### DECISION — Drag included in longitudinal force budget
Combined longitudinal demand = |Δv|/Δt + (c_drag/mass)·v_avg². If this exceeds a_lon_max the transition is penalised.
New parameters: `--mass 230.0 kg`, `--c-drag 0.5 kg/m`.
Default FSAE car values; documented here and in CLAUDE.md.

### DECISION — Speed-coloured path via LineCollection
When `speed_profile` is provided, `plot_track` renders the path as a `LineCollection` coloured by speed (colormap: plasma). Each segment's colour is the midpoint of the two endpoint speeds. A colorbar is added. Falls back to solid red when no speed data is provided.
New import in `visualiser.py`: `matplotlib.collections.LineCollection`.

### DECISION — Stage 5 forward-backward pass simplified
With the physics DP outputting speeds directly, Stage 5 becomes a refinement/validation pass rather than a first-principles derivation. The `(x, y, v)` waypoints come from DP backtracking.

---

## Stage 3 — DP Path Optimiser (2026-04-08)

### STAGE — Stage 3 complete
Deliverable: `python script.py` runs the DP optimiser and plots the raw optimised path (red) overlaid on the track boundaries and naive centerline (orange dashed).
Files added/changed: `optimiser.py` (new), `visualiser.py` (dp_path overlay), `script.py` (wired in).

### DECISION — Extended DP state: (gate, sample, prev_sample)
Standard 2D state (gate, sample) cannot compute exact curvature cost because curvature at a node requires knowing the predecessor. Tracking prev_sample as part of the state gives exact 3-point curvature at every node with O(N_gates × N³) complexity.
For N=11, G=100: ~133 k vectorised operations — effectively instant.
**Alternative considered:** 2D DP with heuristic curvature from the last backtracked path. Rejected: adds a second pass and loses exactness.

### DECISION — Curvature = inverse circumradius (3-point formula)
κ = 4·area / (|AB|·|BC|·|CA|). Returns 0 for collinear points. Fully vectorised over the (Nk, Nj, Nl) index space using numpy broadcasting — no Python loops inside the curvature computation.

### DECISION — Wrap-around curvature not penalised at Stage 3
The DP is a linear pass (gate 0 → gate N-1). The curvature at the seam (gate N-1 → gate 0) is not included in the cost.
**Impact:** slight sub-optimality at the start/finish on closed tracks.
**Deferred to:** Stage 4 (smoothing naturally handles the loop closure).

### FIX — Replaced κ (circumradius inverse) with θ² (heading-change squared)
Root cause: κ has units of 1/length, so for tracks in pixel coordinates the typical value is ~10⁻³. With w_len=1 and w_curve=0.5 the curvature cost was ~184× smaller than the length cost — the algorithm was effectively finding the shortest path only.
θ² = (angle between incoming and outgoing vectors)² is dimensionless, ∈ [0, π²], and scales with neither coordinate units nor gate spacing. With w_curve=5.0 the costs are balanced (ratio ~5×), giving the DP real incentive to prefer smoother arcs.
Confirmed behaviour: on a straight→corner→straight track the path correctly swings wide on the approach, hits the apex, then exits wide (classic racing line). On a pure circular arc, all gate positions have identical θ per gate (same angular step at any radius), so length correctly dominates — the inside arc is genuinely shorter and equally curved.

### DECISION — Default w_curve raised from 0.5 to 5.0
With θ²-based cost, w_curve is now comparable to w_len. 5.0 gives a strong racing-line bias without completely ignoring path length. Users can increase it further for more aggressive cornering or decrease it for a tighter (shorter) path.

### DECISION — DP path verified in-bounds
After optimisation, `max(offset - half_width) ≤ 0` is confirmed numerically. Nodes are constructed by linear interpolation so this holds by construction; the check is a regression guard.

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
