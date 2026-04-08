"""
Smoother module.

Responsibilities:
- Fit a parametric cubic spline through DP waypoints (closed loop)
- Resample path at uniform arc-length intervals
- Check resampled path stays within track boundaries; warn on violations
- Compute signed curvature κ(s) along the smoothed path
"""
