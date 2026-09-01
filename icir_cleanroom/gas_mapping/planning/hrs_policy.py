"""Pure HRS UCB and distance-discount policies."""

import math

import numpy as np


def normalized_ucb(mean, variance, coefficient=1.0):
    """Return a normalized concentration upper-confidence potential."""
    coefficient = float(coefficient)
    if coefficient < 0.0:
        raise ValueError('UCB coefficient must be non-negative')
    mean_values = np.asarray(mean, dtype=float)
    variance_values = np.asarray(variance, dtype=float)
    if mean_values.shape != variance_values.shape:
        raise ValueError('mean and variance shapes must match')
    if np.any(variance_values < -1.0e-12):
        raise ValueError('variance must be non-negative')
    return np.clip(
        mean_values + coefficient * np.sqrt(np.maximum(variance_values, 0.0)),
        0.0, 1.0)


def distance_discounted_ucb(
        mean, variance, distances, coefficient=1.0, distance_weight=0.0):
    """Return UCB discounted by metric distance from the current pose."""
    weight = float(distance_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError('distance weight must be finite and non-negative')
    distance_values = np.asarray(distances, dtype=float)
    potentials = normalized_ucb(mean, variance, coefficient)
    if distance_values.shape != potentials.shape:
        raise ValueError('distances and UCB values must have matching shapes')
    if (np.any(~np.isfinite(distance_values)) or
            np.any(distance_values < 0.0)):
        raise ValueError('distances must be finite and non-negative')
    return potentials / (1.0 + weight * distance_values)


__all__ = [
    'distance_discounted_ucb', 'normalized_ucb',
]
