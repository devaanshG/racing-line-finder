"""
Optimiser module.

Responsibilities:
- Discretise each gate into N_lat candidate positions
- Build a physics-based graph: (gate, lateral_position, velocity)
- Solve for the minimum lap-time path using dynamic programming
- Return the optimal path (N_gates, 2) and speed profile (N_gates,)

Cost function
-------------
Each transition from node (gate_i, lat_j, vel_k) to (gate_{i+1}, lat_l, vel_m)
has a real cost in seconds:

    transition_time = dist(B[j], C[l]) / v_avg(vel_k, vel_m)

Infeasible transitions are penalised with PENALTY (1e6 s each):

  1. Direction penalty  — displacement B[j]→C[l] must point in the forward
     half-plane (dot with gate-to-gate direction > COS_ANGLE_LIMIT).

  2. Lateral accel penalty — v_avg² / R > a_lat_max  (grip limit exceeded).
     R is the circumradius of the triangle A[p]→B[j]→C[l], computed from
     the actual path nodes.  This varies per (p,j,l): outer arc = large R =
     higher corner speed = less time → the racing line emerges naturally.

  3. Longitudinal + drag penalty — |Δv|/Δt + (c_drag/mass)·v_avg² > a_lon_max.

Algorithm
---------
State:   (lat_j, vel_k, prev_lat_p) — 3D DP table per gate, N_lat × N_vel × N_lat.
  dp[j, k, p] = minimum time accumulated to arrive at the current gate
                at lateral position j, speed k, having come from position p.

Tracking prev_lat enables the exact 3-point path curvature A[p]→B[j]→C[l]
without any look-ahead.  The racing line emerges because:
  - Inside hug: small circumradius → high κ → low speed limit → more time
  - Wide arc:   large circumradius → low κ → high speed limit → less time

Transitions per gate: N_lat × N_vel × N_lat × N_vel = N_lat² × N_vel²
Complexity: O(N_gates × N_lat³ × N_vel²)
  At N_lat=N_vel=11, G=100: ~16 M numpy ops, < 1 s.

Backtracking: pred[gate, l, m, j] = (p_prev, k_prev) recovers both the
path and the speed profile.
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
A_LON_DEFAULT:     float = 8.0    # m/s²
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

    # Gate midpoints → forward direction unit vectors (for direction penalty)
    midpoints = (left + right) / 2.0              # (G, 2)
    fwd_vecs  = np.diff(midpoints, axis=0)         # (G-1, 2)
    fwd_norms = np.linalg.norm(fwd_vecs, axis=1, keepdims=True)
    fwd_unit  = fwd_vecs / np.where(fwd_norms > 1e-12, fwd_norms, 1.0)  # (G-1, 2)

    # Fallback curvature from the track centreline — used only for the
    # first gate (i=0) where there is no previous gate to form a 3-point arc.
    kappa_center = np.zeros(G, dtype=np.float64)
    for gi in range(1, G - 1):
        A, B, C = midpoints[gi - 1], midpoints[gi], midpoints[gi + 1]
        AB, BC, AC = B - A, C - B, C - A
        cross = abs(AB[0] * BC[1] - AB[1] * BC[0])
        denom = np.linalg.norm(AB) * np.linalg.norm(BC) * np.linalg.norm(AC)
        kappa_center[gi] = 2.0 * cross / denom if denom > 1e-12 else 0.0
    kappa_center[0]     = kappa_center[1]
    kappa_center[G - 1] = kappa_center[G - 2]

    # Pre-compute velocity-pair products (reused every gate)
    v_avg  = (vel_grid[:, np.newaxis] + vel_grid[np.newaxis, :]) / 2.0   # (k, m)
    a_drag = (c_drag / mass) * v_avg ** 2                                  # (k, m)
    dv     = np.abs(vel_grid[np.newaxis, :] - vel_grid[:, np.newaxis])    # (k, m)

    # ------------------------------------------------------------------
    # Initialise: all states at gate 0 have cost 0.
    # State: dp[j, k, p] — current lat j, current vel k, prev lat p.
    # At the start, prev_lat is fictitious; we allow all combinations.
    # ------------------------------------------------------------------
    dp   = np.zeros((n_lat, n_vel, n_lat), dtype=np.float64)
    # pred[gate, l, m, j] = (p_prev, k_prev)
    #   l, m = chosen lat/vel at this gate
    #   j    = prev_lat (= lat at previous gate)
    #   p_prev = lat two gates back, k_prev = vel at previous gate
    pred = np.full((G, n_lat, n_vel, n_lat, 2), -1, dtype=np.int32)

    # ------------------------------------------------------------------
    # Forward DP: gate i → gate i+1  for i = 0 … G-2
    # ------------------------------------------------------------------
    for i in range(G - 1):
        B = nodes[i]       # (N_lat, 2) — current gate positions
        C = nodes[i + 1]   # (N_lat, 2) — next gate positions

        # --- Segment geometry ----------------------------------------
        # diff_jl[j, l] = vector from B[j] to C[l]
        diff_jl   = C[np.newaxis, :, :] - B[:, np.newaxis, :]   # (j, l, 2)
        dist      = np.linalg.norm(diff_jl, axis=-1)             # (j, l)
        dist_safe = np.where(dist > 1e-12, dist, 1.0)

        # --- Direction penalty ----------------------------------------
        disp_unit = diff_jl / dist_safe[:, :, np.newaxis]        # (j, l, 2)
        dot_fwd   = np.sum(
            disp_unit * fwd_unit[i][np.newaxis, np.newaxis, :], axis=-1
        )                                                          # (j, l)
        dir_pen   = np.where(dot_fwd < COS_ANGLE_LIMIT, PENALTY_DIRECTION, 0.0)

        # --- Transition time -----------------------------------------
        # seg_time[j, l, k, m] = dist[j,l] / v_avg[k,m]
        seg_time  = (dist[:, :, np.newaxis, np.newaxis]
                     / v_avg[np.newaxis, np.newaxis, :, :])       # (j, l, k, m)

        # --- Lateral acceleration penalty ----------------------------
        # For gate i > 0: use the exact circumradius R of the triangle
        #   A[p] = nodes[i-1][p],  B[j] = nodes[i][j],  C[l] = nodes[i+1][l]
        # R[p,j,l] = |AB|·|BC|·|CA| / (2·|AB×BC|)
        # a_lat = v_avg² / R  →  penalise if > a_lat_max
        #
        # This is where the racing line emerges: the outer arc through a
        # corner has a larger R (smaller κ), allowing higher speed, so
        # the DP favours wide-entry → apex → wide-exit paths.
        #
        # For gate i = 0 there is no gate i-1, so fall back to the
        # track centreline curvature (same for all lateral positions).
        if i > 0:
            A = nodes[i - 1]                                           # (N_lat_p, 2)

            AB = B[np.newaxis, :, :] - A[:, np.newaxis, :]            # (p, j, 2)
            AC = C[np.newaxis, :, :] - A[:, np.newaxis, :]            # (p, l, 2)

            AB_len = np.linalg.norm(AB, axis=-1)                      # (p, j)
            AC_len = np.linalg.norm(AC, axis=-1)                      # (p, l)

            # Cross product |AB[p,j] × BC[j,l]|  →  (p, j, l)
            AB_exp = AB[:, :, np.newaxis, :]                           # (p, j, 1, 2)
            BC_exp = diff_jl[np.newaxis, :, :, :]                     # (1, j, l, 2)
            cross  = np.abs(
                AB_exp[:, :, :, 0] * BC_exp[:, :, :, 1]
                - AB_exp[:, :, :, 1] * BC_exp[:, :, :, 0]
            )                                                          # (p, j, l)

            # Circumradius: R = |AB|·|BC|·|AC| / (2·|AB×BC|)
            numer  = (AB_len[:, :, np.newaxis]
                      * dist[np.newaxis, :, :]
                      * AC_len[:, np.newaxis, :])                      # (p, j, l)
            R      = np.where(cross > 1e-12, numer / cross, 1e9)      # large R → straight
            kappa  = 1.0 / R                                           # (p, j, l)

            # a_lat[p, j, l, k, m] = v_avg[k,m]² · κ[p,j,l]
            a_lat_req = (kappa[:, :, :, np.newaxis, np.newaxis]
                         * v_avg[np.newaxis, np.newaxis, np.newaxis, :, :] ** 2)
            lat_pen   = np.where(
                a_lat_req > a_lat_max, PENALTY_FORCE, 0.0
            )                                                          # (p, j, l, k, m)
        else:
            # No previous gate: use track centreline curvature.
            a_lat_req = v_avg ** 2 * kappa_center[i]                  # (k, m)
            lat_pen   = np.where(
                a_lat_req > a_lat_max, PENALTY_FORCE, 0.0
            )[np.newaxis, np.newaxis, np.newaxis, :, :]                # (1, 1, 1, k, m)

        # --- Longitudinal + drag penalty -----------------------------
        a_lon_demand = (dv[np.newaxis, np.newaxis, :, :]
                        / np.where(seg_time > 1e-12, seg_time, 1.0)
                        + a_drag[np.newaxis, np.newaxis, :, :])       # (j, l, k, m)
        lon_pen      = np.where(a_lon_demand > a_lon_max, PENALTY_FORCE, 0.0)

        # --- Total cost tensor  (p, j, l, k, m) ---------------------
        # dp[j, k, p] transposed → (p, j, k) for broadcasting
        dp_t = dp.transpose(2, 0, 1)                                   # (p, j, k)

        total = (
            dp_t[:, :, np.newaxis, :, np.newaxis]                     # (p, j, 1, k, 1)
            + seg_time[np.newaxis, :, :, :, :]                        # (1, j, l, k, m)
            + lat_pen                                                  # (p, j, l, k, m)
            + lon_pen[np.newaxis, :, :, :, :]                         # (1, j, l, k, m)
            + dir_pen[np.newaxis, :, :, np.newaxis, np.newaxis]       # (1, j, l, 1, 1)
        )                                                              # (p, j, l, k, m)

        # --- Minimise over (p, k) → new dp[l, m, j] -----------------
        # j becomes the new prev_lat for the next gate.
        # Transpose to (l, m, j, p, k) → reduce over last 2 dims.
        total_t  = total.transpose(2, 4, 1, 0, 3)                     # (l, m, j, p, k)
        flat     = total_t.reshape(n_lat, n_vel, n_lat, -1)           # (l, m, j, p*k)

        dp       = flat.min(axis=-1)                                   # (l, m, j)
        flat_idx = flat.argmin(axis=-1)                                # (l, m, j)

        pred[i + 1, :, :, :, 0] = flat_idx // n_vel                   # p_prev index
        pred[i + 1, :, :, :, 1] = flat_idx  % n_vel                   # k_prev index

    # ------------------------------------------------------------------
    # Backtrack: recover lat_idx and vel_idx for every gate
    # dp shape at G-1: (N_lat_l, N_vel_m, N_lat_j_prev)
    # ------------------------------------------------------------------
    lat_idx = np.empty(G, dtype=np.int32)
    vel_idx = np.empty(G, dtype=np.int32)

    l, m, j = np.unravel_index(int(np.argmin(dp)), dp.shape)
    lat_idx[G - 1] = l   # lat at final gate
    vel_idx[G - 1] = m   # vel at final gate
    lat_idx[G - 2] = j   # lat at penultimate gate (= prev_lat of final state)

    for gate_i in range(G - 1, 0, -1):
        p = int(pred[gate_i, l, m, j, 0])   # lat two gates back
        k = int(pred[gate_i, l, m, j, 1])   # vel at previous gate
        vel_idx[gate_i - 1] = k
        if gate_i >= 2:
            lat_idx[gate_i - 2] = p
        l, m, j = j, k, p

    path   = np.array([nodes[i, lat_idx[i]] for i in range(G)])
    speeds = vel_grid[vel_idx]

    return path, speeds
