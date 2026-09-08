"""Pure GMRF weighted-centroid policies for HRS target selection."""

import math

import numpy as np


def concentration_weights(mean, threshold):
    """Return threshold-excess weights ``max(mean - threshold, 0)``."""
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError('centroid threshold must be finite and in [0, 1]')
    values = np.asarray(mean, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError('GMRF mean must be finite')
    return np.maximum(values - threshold, 0.0)


def weighted_centroid(positions, weights):
    """Return the weighted x/y centroid, or ``None`` for zero total mass."""
    points = np.asarray(positions, dtype=float)
    values = np.asarray(weights, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError('positions must have shape (N, 2)')
    if values.shape != (points.shape[0],):
        raise ValueError('weights must have shape (N,)')
    if np.any(~np.isfinite(points)) or np.any(~np.isfinite(values)):
        raise ValueError('positions and weights must be finite')
    if np.any(values < 0.0):
        raise ValueError('weights must be non-negative')
    total = float(np.sum(values))
    if total <= np.finfo(float).eps:
        return None
    centroid = np.sum(points * values[:, None], axis=0) / total
    return float(centroid[0]), float(centroid[1])


__all__ = ['concentration_weights', 'weighted_centroid']
