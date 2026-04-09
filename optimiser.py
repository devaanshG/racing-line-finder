"""
Optimiser module.

Responsibilities:
- Discretise each gate into N_lat candidate positions
- Build a 3D node graph: (gate, lateral_position, velocity)
- Solve for the minimum lap-time path using dynamic programming
- Return the optimal path (N_gates, 2) and speed profile (N_gates,)

Cost function
-------------
Each transition from node (gate_i, lat_j, vel_k) to (gate_{i+1}, lat_l, vel_m)
has a real cost in seconds:

    transition_time = dist(B[j], C[l]) / v_avg(vel_k, vel_m)

Infeasible transitions are penalised with PENALTY_DIRECTION / PENALTY_FORCE
(1e6 s each), making them effectively unreachable:

  1. Direction penalty  — displacement B[j]→C[l] must point in the forward
     half-plane (dot with gate-to-gate direction > COS_ANGLE_LIMIT).
     Prevents the path cutting hard across the track against the flow.

  2. Lateral accel penalty — v_avg² · κ > a_lat_max  (grip limit exceeded).
     κ is approximated from the cross-track angle of the displacement vector
     relative to the forward gate direction.

  3. Longitudinal penalty — (|Δv| / Δt) + (c_drag/mass)·v_avg² > a_lon_max
     (combined engine/braking + drag demand exceeds capability).

Algorithm
---------
State:   (lat_j, vel_k) — 2D DP table per gate, size N_lat × N_vel.
  dp[j, k] = minimum time accumulated to arrive at the current gate
             at lateral position j with speed k.

Forward DP recurrence (fully vectorised, no Python loop over transitions):
  total[j, l, k, m] = dp[j, k] + seg_time[j,l,k,m] + penalties
  dp_new[l, m]      = min over (j, k) of total[j, l, k, m]

Complexity: O(N_gates × N_lat² × N_vel²)
  At N_lat = N_vel = 11, G = 100: ~1.43 M numpy ops, < 50 ms.

Backtracking: pred[gate, l, m] = (prev_j, prev_k) recovers both the
path and the speed profile in a single pass.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Module-level defaults  (mirrored as CLI args in script.py)
# ---------------------------------------------------------------------------
N_LAT_DEFAULT:     int   = 11
N_VEL_DEFAULT:     int   = 11
V_MIN_DEFAULT:     float = 1.0    # m/s — floor of velocity grid; must be > 0
V_MAX_DEFAULT:     float = 15.0   # m/s
A_LAT_DEFAULT:     float = 12.0   # m/s²
A_LON_DEFAULT:     float = 8.0    # m/s²  (combined engine + braking capability)
MASS_DEFAULT:      float = 230.0  # kg    (typical FSAE car incl. driver)
C_DRAG_DEFAULT:    float = 0.5    # kg/m  (aerodynamic drag: c_d × frontal area)

PENALTY_DIRECTION: float = 1e6    # seconds — direction-alignment violation
PENALTY_FORCE:     float = 1e6    # seconds — force-budget violation
COS_ANGLE_LIMIT:   float = 0.0    # displacement must have dot(disp, fwd) > this


# ---------------------------------------------------------------------------
# Gate discretisation
# ---------------------------------------------------------------------------

def sample_gates(left: np.ndarray, right: np.ndarray, n: int) -> np.ndarray:
    """
    Discretise each gate into n evenly-spaced candidate positions.

    j=0   ↔ left cone
    j=n-1 ↔ right cone

    Returns
    -------
    nodes : np.ndarray (N_gates, n, 2)
    """
    t    = np.linspace(0.0, 1.0, n)
    diff = (right - left)[:, np.newaxis, :]
    return left[:, np.newaxis, :] + t[np.newaxis, :, np.newaxis] * diff


# ---------------------------------------------------------------------------
# DP solver
# ---------------------------------------------------------------------------

def optimise(
    left:      np.ndarray,
    right:     np.ndarray,
    n_lat:     int   = N_LAT_DEFAULT,
    n_vel:     int   = N_VEL_DEFAULT,
    v_max:     float = V_MAX_DEFAULT,
    v_min:     float = V_MIN_DEFAULT,
    a_lat_max: float = A_LAT_DEFAULT,
    a_lon_max: float = A_LON_DEFAULT,
    mass:      float = MASS_DEFAULT,
    c_drag:    float = C_DRAG_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the minimum lap-time path through the track gates.

    Parameters
    ----------
    left, right : np.ndarray (N_gates, 2)
    n_lat       : lateral position samples per gate
    n_vel       : velocity grid resolution
    v_max       : maximum speed (m/s)
    v_min       : minimum speed on velocity grid (m/s) — must be > 0
    a_lat_max   : max lateral acceleration (m/s²)
    a_lon_max   : max longitudinal accel/braking demand (m/s²)
    mass        : vehicle mass (kg)
    c_drag      : drag coefficient × frontal area (kg/m)

    Returns
    -------
    path   : np.ndarray (N_gates, 2)  — optimised position at each gate
    speeds : np.ndarray (N_gates,)    — speed in m/s at each gate
    """
    nodes = sample_gates(left, right, n_lat)   # (G, N_lat, 2)
    G, _, _ = nodes.shape

    if G < 3:
        raise ValueError("Need at least 3 gates to run DP.")

    vel_grid = np.linspace(v_min, v_max, n_vel)   # (N_vel,)

    # Pre-compute gate-to-gate forward unit vectors (for direction penalty)
    midpoints = (left + right) / 2.0              # (G, 2)
    fwd_vecs  = np.diff(midpoints, axis=0)         # (G-1, 2)
    fwd_norms = np.linalg.norm(fwd_vecs, axis=1, keepdims=True)
    fwd_unit  = fwd_vecs / np.where(fwd_norms > 1e-12, fwd_norms, 1.0)  # (G-1, 2)

    # Pre-compute track curvature κ at each gate from the centerline midpoints.
    # κ[i] = 2·|AB × BC| / (|AB|·|BC|·|AC|)  — inverse circumradius of A, B, C.
    # This is the curvature the car must negotiate regardless of its lateral
    # position within the gate.  Indices: κ is defined for gates 1..G-2;
    # gates 0 and G-1 use their neighbours' values (boundary extrapolation).
    kappa_gate = np.zeros(G, dtype=np.float64)
    for gi in range(1, G - 1):
        A, B, C = midpoints[gi - 1], midpoints[gi], midpoints[gi + 1]
        AB = B - A;  BC = C - B;  AC = C - A
        cross = abs(AB[0] * BC[1] - AB[1] * BC[0])
        denom = np.linalg.norm(AB) * np.linalg.norm(BC) * np.linalg.norm(AC)
        kappa_gate[gi] = 2.0 * cross / denom if denom > 1e-12 else 0.0
    kappa_gate[0]     = kappa_gate[1]
    kappa_gate[G - 1] = kappa_gate[G - 2]

    # v_avg[k, m] = average speed of a transition from vel k to vel m
    v_avg = (vel_grid[:, np.newaxis] + vel_grid[np.newaxis, :]) / 2.0   # (N_vel, N_vel)

    # Longitudinal drag deceleration at each speed pair: a_drag = (c_drag/mass) * v_avg²
    a_drag = (c_drag / mass) * v_avg ** 2          # (N_vel, N_vel)

    # |Δv| between every pair of velocity grid points
    dv = np.abs(vel_grid[np.newaxis, :] - vel_grid[:, np.newaxis])       # (N_vel, N_vel)

    # ------------------------------------------------------------------
    # Initialise: all nodes at gate 0 are reachable at cost 0.
    # The car can start anywhere on the first gate at any speed.
    # ------------------------------------------------------------------
    dp   = np.zeros((n_lat, n_vel), dtype=np.float64)
    pred = np.full((G, n_lat, n_vel, 2), -1, dtype=np.int32)

    # ------------------------------------------------------------------
    # Forward DP: gate i → gate i+1  for i = 0 … G-2
    # ------------------------------------------------------------------
    for i in range(G - 1):
        B = nodes[i]       # (N_lat, 2) — current gate positions
        C = nodes[i + 1]   # (N_lat, 2) — next gate positions

        # --- Segment geometry ----------------------------------------
        # diff_jl[j, l] = vector from B[j] to C[l]
        diff_jl = C[np.newaxis, :, :] - B[:, np.newaxis, :]   # (N_lat_j, N_lat_l, 2)
        dist    = np.linalg.norm(diff_jl, axis=-1)             # (N_lat_j, N_lat_l)
        dist_safe = np.where(dist > 1e-12, dist, 1.0)

        # --- Direction penalty ----------------------------------------
        # Displacement unit vector for each (j→l) pair
        disp_unit = diff_jl / dist_safe[:, :, np.newaxis]      # (N_lat_j, N_lat_l, 2)
        dot_fwd   = np.sum(
            disp_unit * fwd_unit[i][np.newaxis, np.newaxis, :], axis=-1
        )                                                        # (N_lat_j, N_lat_l)
        dir_pen = np.where(dot_fwd < COS_ANGLE_LIMIT, PENALTY_DIRECTION, 0.0)

        # --- Transition time -----------------------------------------
        # seg_time[j, l, k, m] = dist[j,l] / v_avg[k,m]
        seg_time = (dist[:, :, np.newaxis, np.newaxis]
                    / v_avg[np.newaxis, np.newaxis, :, :])      # (j, l, k, m)

        # --- Lateral acceleration penalty ----------------------------
        # Use the pre-computed track curvature κ at the current gate.
        # The centripetal acceleration v²·κ must not exceed a_lat_max.
        # κ is the same for all (j, l) pairs at a given gate — it measures
        # the curvature of the track itself at this location.
        a_lat_req = v_avg ** 2 * kappa_gate[i]                  # (N_vel_k, N_vel_m)
        lat_pen   = np.where(a_lat_req > a_lat_max, PENALTY_FORCE, 0.0)
        # Broadcast to (j, l, k, m) — same penalty for all lateral pairs at this gate
        lat_pen   = lat_pen[np.newaxis, np.newaxis, :, :]

        # --- Longitudinal + drag penalty -----------------------------
        # Combined demand: acceleration needed for Δv + drag resistance
        #   a_lon_demand = |Δv| / seg_time + (c_drag/mass) · v_avg²
        # If this exceeds a_lon_max (engine/braking capability), penalise.
        a_lon_demand = (dv[np.newaxis, np.newaxis, :, :]
                        / np.where(seg_time > 1e-12, seg_time, 1.0)
                        + a_drag[np.newaxis, np.newaxis, :, :]) # (j, l, k, m)
        lon_pen = np.where(a_lon_demand > a_lon_max, PENALTY_FORCE, 0.0)

        # --- Total cost tensor ---------------------------------------
        # total[j, l, k, m] = dp[j,k] + seg_time + all penalties
        total = (dp[:, np.newaxis, :, np.newaxis]
                 + seg_time
                 + dir_pen[:, :, np.newaxis, np.newaxis]
                 + lat_pen
                 + lon_pen)

        # --- Minimise over (j, k) → new dp[l, m] --------------------
        # Transpose to (N_lat_l, N_vel_m, N_lat_j, N_vel_k) so we can
        # argmin over the last two axes, which correspond to (j, k).
        total_t  = total.transpose(1, 3, 0, 2)
        flat     = total_t.reshape(n_lat, n_vel, -1)
        dp       = flat.min(axis=-1)                            # (N_lat_l, N_vel_m)
        flat_idx = flat.argmin(axis=-1)                         # (N_lat_l, N_vel_m)

        pred[i + 1, :, :, 0] = flat_idx // n_vel               # best prev_lat for each (l, m)
        pred[i + 1, :, :, 1] = flat_idx  % n_vel               # best prev_vel for each (l, m)

    # ------------------------------------------------------------------
    # Backtrack: walk pred to recover lat and vel index at every gate
    # ------------------------------------------------------------------
    lat_idx = np.empty(G, dtype=np.int32)
    vel_idx = np.empty(G, dtype=np.int32)

    l_curr, m_curr = np.unravel_index(int(np.argmin(dp)), dp.shape)
    lat_idx[G - 1] = l_curr
    vel_idx[G - 1] = m_curr

    for i in range(G - 1, 0, -1):
        j_prev = int(pred[i, l_curr, m_curr, 0])
        k_prev = int(pred[i, l_curr, m_curr, 1])
        lat_idx[i - 1] = j_prev
        vel_idx[i - 1] = k_prev
        l_curr, m_curr = j_prev, k_prev

    path   = np.array([nodes[i, lat_idx[i]] for i in range(G)])
    speeds = vel_grid[vel_idx]

    return path, speeds
