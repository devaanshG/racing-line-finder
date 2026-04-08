"""
Optimiser module.

Responsibilities:
- Discretise each gate into N candidate positions
- Build a directed graph of nodes across gates
- Solve for the minimum-cost path using dynamic programming
  Cost = w_len * segment_length + w_curve * curvature_penalty
- Return the optimal path as an (N_gates, 2) coordinate array

Algorithm
---------
State space: (gate_i, sample_j, prev_sample_k)
  At gate i, the car is at sample j having come from sample k at gate i-1.
  Tracking the previous sample enables an exact curvature cost at each node
  using the three-point circumradius formula, with no look-ahead required.

DP recurrence (vectorised):
  dp[l, j]  = min_k ( dp[j, k]
                     + w_len  * dist(B[j], C[l])
                     + w_curve * curv(A[k], B[j], C[l]) )
  where A = nodes[i-1], B = nodes[i], C = nodes[i+1]

Complexity: O(N_gates × N³)  — N=11 → ~133 k ops, negligible runtime.

Closed-loop note: wrap-around curvature (gate N-1 → gate 0) is not penalised
at this stage.  The path start/end will be slightly sub-optimal on closed
tracks; this is acceptable for Stage 3.
"""

import numpy as np

# Module-level defaults (also used as CLI defaults in script.py)
N_SAMPLES_DEFAULT: int   = 11
W_LEN_DEFAULT:     float = 1.0
W_CURVE_DEFAULT:   float = 0.5


# ---------------------------------------------------------------------------
# Gate discretisation
# ---------------------------------------------------------------------------

def sample_gates(left: np.ndarray, right: np.ndarray, n: int) -> np.ndarray:
    """
    Discretise each gate into n evenly-spaced candidate positions.

    Sample j=0 is the left cone; j=n-1 is the right cone.

    Returns
    -------
    nodes : np.ndarray of shape (N_gates, n, 2)
    """
    t = np.linspace(0.0, 1.0, n)                            # (n,)
    diff = (right - left)[:, np.newaxis, :]                  # (N_gates, 1, 2)
    return left[:, np.newaxis, :] + t[np.newaxis, :, np.newaxis] * diff


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------

def _curvature_tensor(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Inverse circumradius (curvature) for all triples (A[k], B[j], C[l]).

    Uses the 3-point formula:  κ = 4·area / (|AB|·|AC|·|BC|)
    Returns 0 for collinear or degenerate triples.

    Parameters
    ----------
    A, B, C : (N, 2)

    Returns
    -------
    curv : (N, N, N) — curv[k, j, l] = κ at B[j] given A[k] and C[l]
    """
    AB = B[np.newaxis, :, :] - A[:, np.newaxis, :]   # (Nk, Nj, 2)
    AC = C[np.newaxis, :, :] - A[:, np.newaxis, :]   # (Nk, Nl, 2)
    BC = C[np.newaxis, :, :] - B[:, np.newaxis, :]   # (Nj, Nl, 2)

    # Signed 2-D cross product → triangle area
    AB_exp = AB[:, :, np.newaxis, :]   # (Nk, Nj, 1,  2)
    AC_exp = AC[:, np.newaxis, :, :]   # (Nk, 1,  Nl, 2)
    cross  = AB_exp[..., 0] * AC_exp[..., 1] - AB_exp[..., 1] * AC_exp[..., 0]
    area   = 0.5 * np.abs(cross)       # (Nk, Nj, Nl)

    c = np.linalg.norm(AB, axis=-1)    # (Nk, Nj)  |A→B|
    b = np.linalg.norm(AC, axis=-1)    # (Nk, Nl)  |A→C|
    a = np.linalg.norm(BC, axis=-1)    # (Nj, Nl)  |B→C|

    # Broadcast side lengths to (Nk, Nj, Nl)
    prod = (a[np.newaxis, :, :]
            * b[:, np.newaxis, :]
            * c[:, :, np.newaxis])

    return np.where(prod > 1e-12, 4.0 * area / prod, 0.0)


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
    Find the minimum-cost path through the track using dynamic programming.

    Parameters
    ----------
    left, right : np.ndarray (N_gates, 2)
    n_samples   : candidate positions per gate  (N)
    w_len       : weight for segment-length cost
    w_curve     : weight for curvature penalty

    Returns
    -------
    path : np.ndarray (N_gates, 2) — one optimised point per gate
    """
    nodes = sample_gates(left, right, n_samples)
    G, N, _ = nodes.shape   # G = N_gates, N = n_samples

    if G < 3:
        raise ValueError("Need at least 3 gates to run DP.")

    # ------------------------------------------------------------------
    # Initialise: gate 0 → gate 1  (length cost only, no curvature)
    # dp[curr, prev] at gate 1
    # ------------------------------------------------------------------
    d01 = np.linalg.norm(
        nodes[1][:, np.newaxis, :] - nodes[0][np.newaxis, :, :], axis=-1
    )                                  # (N_curr, N_prev)
    dp = w_len * d01                   # (N, N)

    # pred[i, l, j] = k at gate i-2 that minimised dp when arriving at
    # state (curr=l, prev=j) at gate i.
    pred = np.full((G, N, N), -1, dtype=np.int32)

    # ------------------------------------------------------------------
    # Forward DP: iterate from gate 1 up to gate G-2
    # Each iteration transitions dp from gate i → gate i+1
    # ------------------------------------------------------------------
    for i in range(1, G - 1):
        A = nodes[i - 1]   # (N, 2)
        B = nodes[i]       # (N, 2)
        C = nodes[i + 1]   # (N, 2)

        dist_jl = np.linalg.norm(
            C[np.newaxis, :, :] - B[:, np.newaxis, :], axis=-1
        )                              # (Nj, Nl)
        curv = _curvature_tensor(A, B, C)  # (Nk, Nj, Nl)

        # total[k, j, l] = dp[j, k] + w_len·dist_jl[j,l] + w_curve·curv[k,j,l]
        total = (dp.T[:, :, np.newaxis]
                 + w_len   * dist_jl[np.newaxis, :, :]
                 + w_curve * curv)     # (Nk, Nj, Nl)

        # Minimise over k  →  new dp[l, j] and pred[i+1][l, j]
        dp          = total.min(axis=0).T     # (Nl, Nj)
        pred[i + 1] = total.argmin(axis=0).T  # (Nl, Nj)

    # ------------------------------------------------------------------
    # Backtrack to recover sample indices
    # ------------------------------------------------------------------
    path_idx = np.empty(G, dtype=np.int32)
    curr, prev = np.unravel_index(int(np.argmin(dp)), dp.shape)
    path_idx[G - 1] = curr
    path_idx[G - 2] = prev

    for i in range(G - 1, 1, -1):
        pp = int(pred[i, curr, prev])
        path_idx[i - 2] = pp
        curr, prev = prev, pp

    path = np.array([nodes[i, path_idx[i]] for i in range(G)])
    return path
