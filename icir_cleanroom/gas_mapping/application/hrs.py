"""HRS runtime state and GMRF weighted-centroid target selection."""

from dataclasses import dataclass
import math

from ..models import HrsRuntimeState
from ..planning.hrs_policy import concentration_weights, weighted_centroid


@dataclass(frozen=True)
class HrsTarget:
    variable: int
    row: int
    col: int
    x: float
    y: float
    mean: float
    weight: float
    centroid_distance: float


@dataclass(frozen=True)
class CentroidSelection:
    centroid: tuple | None
    weights: tuple
    target: HrsTarget | None


class HrsManager:
    def __init__(self, state=None):
        self.state = state or HrsRuntimeState()

    def reset_search(self):
        self.state.reset_search()

    def available_variables(
            self, variable_count, sampled_variables, eligible_variables=None):
        eligible = (set(range(int(variable_count)))
                    if eligible_variables is None
                    else set(eligible_variables))
        return (eligible - set(sampled_variables) -
                self.state.unreachable_variables)

    def record_failure(self, variable, max_failures):
        variable = int(variable)
        count = self.state.failure_counts.get(variable, 0) + 1
        self.state.failure_counts[variable] = count
        unreachable = count >= int(max_failures)
        if unreachable:
            self.state.unreachable_variables.add(variable)
        return count, unreachable

    def record_success(self, variable):
        self.state.failure_counts.pop(int(variable), None)

    def select_target(
            self, gmrf, sampled_variables, threshold,
            eligible_variables=None):
        """Compute the field centroid and project it to one valid HRS cell."""
        weights = concentration_weights(gmrf.solution, threshold)
        positions = tuple(
            gmrf.cell_center(variable)
            for variable in range(len(gmrf.var_cells)))
        centroid = weighted_centroid(positions, weights)
        if centroid is None:
            return CentroidSelection(None, tuple(weights), None)

        available = sorted(self.available_variables(
            len(gmrf.var_cells), sampled_variables, eligible_variables))
        if not available:
            return CentroidSelection(centroid, tuple(weights), None)

        cx, cy = centroid

        def target_key(variable):
            row, col = gmrf.var_cells[variable]
            x, y = gmrf.cell_center(variable)
            return (
                round(math.hypot(x - cx, y - cy), 12),
                int(row), int(col), int(variable))

        variable = min(available, key=target_key)
        row, col = gmrf.var_cells[variable]
        x, y = gmrf.cell_center(variable)
        target = HrsTarget(
            variable=int(variable), row=int(row), col=int(col),
            x=float(x), y=float(y), mean=float(gmrf.solution[variable]),
            weight=float(weights[variable]),
            centroid_distance=math.hypot(x - cx, y - cy))
        return CentroidSelection(centroid, tuple(weights), target)

    @staticmethod
    def reached_response_threshold(value, threshold):
        value = float(value)
        threshold = float(threshold)
        if not math.isfinite(value):
            raise ValueError('measured value must be finite')
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError('response threshold must be in [0, 1]')
        return value >= threshold


__all__ = ['CentroidSelection', 'HrsManager', 'HrsTarget']
