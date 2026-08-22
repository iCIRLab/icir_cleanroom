"""Pure HRS candidate, stopping, and measured-peak policies."""

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


def peak_search_status(mean, variance, available_variables, best_observed,
                       coefficient=1.0, margin=0.02):
    """Evaluate whether any available cell can improve the measured peak."""
    margin = float(margin)
    if margin < 0.0:
        raise ValueError('stop margin must be non-negative')
    variables = np.asarray(sorted(available_variables), dtype=np.int64)
    if variables.size == 0:
        return True, None, None, None
    potentials = normalized_ucb(
        np.asarray(mean)[variables], np.asarray(variance)[variables],
        coefficient)
    local_index = int(np.argmax(potentials))
    variable = int(variables[local_index])
    maximum = float(potentials[local_index])
    gap = maximum - float(best_observed)
    return maximum <= float(best_observed) + margin, variable, maximum, gap


def higher_measured_neighbor(anchor_variable, neighbor_variables,
                             measured_values, variable_cells,
                             improvement_epsilon=0.001):
    """Return a deterministically selected, meaningfully higher neighbor."""
    epsilon = float(improvement_epsilon)
    if epsilon <= 0.0:
        raise ValueError('improvement_epsilon must be positive')
    anchor = int(anchor_variable)
    if anchor not in measured_values:
        raise ValueError('anchor variable has no measured value')
    neighbors = [int(variable) for variable in neighbor_variables]
    missing = [variable for variable in neighbors
               if variable not in measured_values]
    if missing:
        raise ValueError(f'neighbor variables are unmeasured: {missing}')
    cells = np.asarray(variable_cells)
    best = min(
        [anchor] + neighbors,
        key=lambda variable: (
            -float(measured_values[variable]),
            int(cells[variable][0]), int(cells[variable][1])))
    if (best != anchor and
            float(measured_values[best]) >=
            float(measured_values[anchor]) + epsilon):
        return best
    return None


__all__ = [
    'higher_measured_neighbor', 'normalized_ucb', 'peak_search_status',
]
