"""
Optimiser module.

Responsibilities:
- Discretise each gate into N candidate positions
- Build a directed graph of nodes across gates
- Solve for the minimum-cost path using dynamic programming
  Cost = w_len * segment_length + w_curve * curvature_penalty
"""
