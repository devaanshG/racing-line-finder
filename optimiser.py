"""
Optimiser module.

Responsibilities:
- Discretise each gate into N candidate positions
- Build a directed graph of nodes across gates
- Solve for the minimum-cost path using dynamic programming
- Return the optimal path as an (N_gates, 2) coordinate array

Cost function
-------------
  cost = w_len * segment_length + w_curve * heading_change²

Heading change θ at node B is the angle between the incoming vector (A→B)
and the outgoing vector (B→C), computed as arccos of the normalised dot
product.  θ ∈ [0, π] rad.

Why θ² and not κ (circumradius inverse)?
  κ is in units of 1/length and scales with the coordinate system; for a
  track digitised in pixels, κ is tiny (~1e-3) and is swamped by the length
  term regardless of w_curve.  θ² is always dimensionless, ranging from 0
  (straight) to π² (~9.87) for a full U-turn, making w_len and w_curve
  directly comparable.  The quadratic also more aggressively penalises sharp
  corners, naturally producing the wide-entry → apex → wide-exit racing line.

Algorithm
---------
State space: (gate_i, sample_j, prev_sample_k)
  Tracking the previous sample allows the heading-change cost to be computed
  exactly at every node with no look-ahead.

DP recurrence (vectorised over k, j, l):
  dp[l, j]  = min_k ( dp[j, k]
                     + w_len  * dist(B[j], C[l])
                     + w_curve * θ²(A[k]→B[j]→C[l]) )
  A = nodes[i-1], B = nodes[i], C = nodes[i+1]

Complexity: O(N_gates × N³) — N=11 → ~133 k numpy ops, near-instant.

Closed-loop note: wrap-around curvature (gate N-1 → gate 0) is not penalised
here; handled naturally by the spline smoother in Stage 4.
"""

import numpy as np

# Module-level defaults (mirrored in script.py CLI)
N_SAMPLES_DEFAULT: int   = 11
W_LEN_DEFAULT:     float = 1.0
W_CURVE_DEFAULT:   float = 5.0   # θ² is dimensionless; 5.0 gives strong racing-line bias


# ---------------------------------------------------------------------------
# Gate discretisation
# ---------------------------------------------------------------------------

def sample_gates(left: np.ndarray, right: np.ndarray, n: int) -> np.ndarray:
    """
    Discretise each gate into n evenly-spaced candidate positions.

    j=0  ↔  left cone
    j=n-1 ↔ right cone

    Returns
    -------
    nodes : np.ndarray (N_gates, n, 2)
    """
    t    = np.linspace(0.0, 1.0, n)
    diff = (right - left)[:, np.newaxis, :]                     # (G, 1, 2)
    return left[:, np.newaxis, :] + t[np.newaxis, :, np.newaxis] * diff


# ---------------------------------------------------------------------------
# Heading-change cost tensor
# ---------------------------------------------------------------------------

def _heading_change_sq(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Squared heading change θ² at B for every triple (A[k], B[j], C[l]).

    θ = angle between vectors (A[k]→B[j]) and (B[j]→C[l])
      = arccos( (A→B · B→C) / (|A→B| · |B→C|) )

    Returns 0 when either segment has zero length (degenerate gate).

    Parameters
    ----------
    A, B, C : (N, 2)

    Returns
    -------
    theta_sq : (N, N, N) — theta_sq[k, j, l] = θ² at B[j]
    """
    v1 = B[np.newaxis, :, :] - A[:, np.newaxis, :]     # (Nk, Nj, 2)  A→B
    v2 = C[np.newaxis, :, :] - B[:, np.newaxis, :]     # (Nj, Nl, 2)  B→C

    # Expand to (Nk, Nj, Nl, 2) for broadcasting
    v1_exp = v1[:, :, np.newaxis, :]                    # (Nk, Nj,  1, 2)
    v2_exp = v2[np.newaxis, :, :, :]                    # ( 1, Nj, Nl, 2)

    dot    = np.sum(v1_exp * v2_exp, axis=-1)           # (Nk, Nj, Nl)
    norm1  = np.linalg.norm(v1, axis=-1)[:, :, np.newaxis]   # (Nk, Nj, 1)
    norm2  = np.linalg.norm(v2, axis=-1)[np.newaxis, :, :]   # ( 1, Nj, Nl)
    prod   = norm1 * norm2                              # (Nk, Nj, Nl)

    cos_t  = np.where(prod > 1e-12, dot / prod, 1.0)   # 0 penalty when degenerate
    cos_t  = np.clip(cos_t, -1.0, 1.0)
    theta  = np.arccos(cos_t)                           # (Nk, Nj, Nl) ∈ [0, π]
    return theta ** 2


# ---------------------------------------------------------------------------
# DP solver
# ---------------------------------------------------------------------------

def optimise(
    left:      np.ndarray,
    right:     np.ndarray,
    n_samples: int   = N_SAMPLES_DEFAULT,
    w_len:     float = W_LEN_DEFAULT,
    w_curve:   float = W_CURVE_DEFAULT,
) -> np.ndarray:
    """
    Find the minimum-cost path through the track gates using dynamic programming.

    Parameters
    ----------
    left, right : np.ndarray (N_gates, 2)
    n_samples   : candidate positions per gate  (N)
    w_len       : weight for segment-length cost
    w_curve     : weight for θ² heading-change penalty
                  Increase to get a smoother, wider arc (racing line).
                  Decrease to get a shorter but tighter path.

    Returns
    -------
    path : np.ndarray (N_gates, 2) — one optimised point per gate
    """
    nodes = sample_gates(left, right, n_samples)
    G, N, _ = nodes.shape

    if G < 3:
        raise ValueError("Need at least 3 gates to run DP.")

    # ------------------------------------------------------------------
    # Initialise: gate 0 → gate 1  (length only; no heading yet)
    # dp[curr, prev] = cost to be at gate 1 sample curr, from gate 0 sample prev
    # ------------------------------------------------------------------
    d01 = np.linalg.norm(
        nodes[1][:, np.newaxis, :] - nodes[0][np.newaxis, :, :], axis=-1
    )                                       # (N_curr, N_prev)
    dp   = w_len * d01                      # (N, N)
    pred = np.full((G, N, N), -1, dtype=np.int32)

    # ------------------------------------------------------------------
    # Forward DP: gate 1 … G-2  (transitions to gate 2 … G-1)
    # ------------------------------------------------------------------
    for i in range(1, G - 1):
        A = nodes[i - 1]   # (N, 2)
        B = nodes[i]       # (N, 2)
        C = nodes[i + 1]   # (N, 2)

        dist_jl  = np.linalg.norm(
            C[np.newaxis, :, :] - B[:, np.newaxis, :], axis=-1
        )                                   # (Nj, Nl)
        theta_sq = _heading_change_sq(A, B, C)  # (Nk, Nj, Nl)

        # total[k, j, l] = dp[j,k] + w_len·dist[j,l] + w_curve·θ²[k,j,l]
        total = (dp.T[:, :, np.newaxis]
                 + w_len   * dist_jl[np.newaxis, :, :]
                 + w_curve * theta_sq)      # (Nk, Nj, Nl)

        dp          = total.min(axis=0).T      # (Nl, Nj)
        pred[i + 1] = total.argmin(axis=0).T  # (Nl, Nj)

    # ------------------------------------------------------------------
    # Backtrack
    # ------------------------------------------------------------------
    path_idx = np.empty(G, dtype=np.int32)
    curr, prev = np.unravel_index(int(np.argmin(dp)), dp.shape)
    path_idx[G - 1] = curr
    path_idx[G - 2] = prev

    for i in range(G - 1, 1, -1):
        pp             = int(pred[i, curr, prev])
        path_idx[i-2]  = pp
        curr, prev     = prev, pp

    return np.array([nodes[i, path_idx[i]] for i in range(G)])
