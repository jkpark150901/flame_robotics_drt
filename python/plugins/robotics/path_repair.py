"""Deterministic dense joint interpolation audit and bounded local replanning.

This is sampled geometric verification, not continuous collision detection.
Callbacks must use the same scene and collision policy as execution.
"""
import numpy as np


def audit_and_repair(path, collision, replan, resolution=0.005, max_repairs=3):
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise RuntimeError('simulation audit: invalid path')
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('audit resolution must be positive')
    points = points.copy()
    history = []
    for attempt in range(max_repairs + 1):
        unsafe = [bool(collision(q)) for q in points]
        if unsafe[0] or unsafe[-1]:
            raise RuntimeError('simulation audit: start/goal collision; cannot preserve endpoints')
        bad = []
        for i, (a, b) in enumerate(zip(points, points[1:])):
            steps = max(1, int(np.ceil(np.linalg.norm(b - a) / resolution)))
            if unsafe[i] or unsafe[i + 1] or any(
                    collision(a + (b - a) * (j / steps)) for j in range(1, steps)):
                bad.append(i)
        if not bad:
            return points, {'resolution': resolution, 'repairs': history, 'colliding_edges': 0}
        if attempt == max_repairs:
            raise RuntimeError(f'simulation audit: collision remains after {max_repairs} repairs')
        # Enclose all bad edges; widen the safe anchors on each retry.
        left, right = max(0, bad[0] - attempt - 1), min(len(points) - 1, bad[-1] + attempt + 2)
        while unsafe[left] and left > 0:
            left -= 1
        while unsafe[right] and right < len(points) - 1:
            right += 1
        replacement = np.asarray(replan(points[left].copy(), points[right].copy(), resolution), dtype=float)
        if (replacement.ndim != 2 or len(replacement) < 2 or replacement.shape[1] != points.shape[1]
                or not np.isfinite(replacement).all()
                or not np.allclose(replacement[0], points[left], atol=1e-6, rtol=0)
                or not np.allclose(replacement[-1], points[right], atol=1e-6, rtol=0)):
            raise RuntimeError('simulation audit: replan failed to connect safe anchors')
        history.append({'bad_edges': bad, 'left': left, 'right': right})
        points = np.concatenate((points[:left], replacement, points[right + 1:]))
