"""Pure HRS UCB and distance-aware score policies."""

import numpy as np


def _validated_fields(mean, variance):
    mean_values = np.asarray(mean, dtype=float)
    variance_values = np.asarray(variance, dtype=float)
    if mean_values.shape != variance_values.shape:
        raise ValueError('mean and variance shapes must match')
    if np.any(~np.isfinite(mean_values)):
        raise ValueError('mean must be finite')
    if np.any(~np.isfinite(variance_values)):
        raise ValueError('variance must be finite')
    if np.any(variance_values < -1.0e-12):
        raise ValueError('variance must be non-negative')
    return mean_values, variance_values


def normalized_ucb(mean, variance, coefficient=1.0):
    """Return unclipped UCB in normalized concentration units."""
    coefficient = float(coefficient)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError('UCB coefficient must be finite and non-negative')
    mean_values, variance_values = _validated_fields(mean, variance)
    return mean_values + coefficient * np.sqrt(np.maximum(variance_values, 0.0))


def _minmax(values, constant_value):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    span = float(np.max(values) - np.min(values))
    if span <= np.finfo(float).eps:
        return np.full(values.shape, float(constant_value), dtype=float)
    return (values - np.min(values)) / span


def distance_aware_scores(ucb, distances, distance_weight):
    """Return DD-UCB scores, raw UCB, and candidate-normalized distances."""
    ucb_values = np.asarray(ucb, dtype=float)
    distance_values = np.asarray(distances, dtype=float)
    if ucb_values.shape != distance_values.shape:
        raise ValueError('UCB and distance shapes must match')
    if np.any(~np.isfinite(ucb_values)):
        raise ValueError('UCB values must be finite')
    if (np.any(~np.isfinite(distance_values)) or
            np.any(distance_values < 0.0)):
        raise ValueError('distances must be finite and non-negative')
    weight = float(distance_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(
            'distance weight must be finite and non-negative')

    normalized_distances = _minmax(distance_values, constant_value=0.0)
    scores = ucb_values - weight * normalized_distances
    return scores, ucb_values, normalized_distances


__all__ = [
    'distance_aware_scores', 'normalized_ucb',
]
