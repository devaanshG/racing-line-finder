"""
Speed profile module.

Responsibilities:
- Compute curvature-limited speed: v_max(s) = sqrt(a_lat / |κ(s)|), clamped to v_max_global
- Forward pass: apply longitudinal acceleration limit
- Backward pass: apply braking limit
- Return array of (x, y, v) waypoints
"""
