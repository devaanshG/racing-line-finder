# CHANGELOG

Records all decisions, stage completions, and notable changes in the Track Path Optimiser project.
Format: `[YYYY-MM-DD] Type — Description`

Types: `DECISION`, `STAGE`, `FIX`, `DEPENDENCY`, `ASSUMPTION`

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
