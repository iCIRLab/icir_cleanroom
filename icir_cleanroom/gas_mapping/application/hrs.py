"""HRS runtime state and single-cell DD-UCB candidate selection."""

from dataclasses import dataclass, replace
import math

from ..models import HrsRuntimeState
from ..planning.hrs_policy import distance_aware_scores, normalized_ucb


@dataclass(frozen=True)
class HrsCandidate:
    variable: int
    row: int
    col: int
    x: float
    y: float
    score: float
    mean: float
    variance: float
    ucb: float | None = None
    distance: float = 0.0
    normalized_ucb: float = 0.0
    normalized_distance: float = 0.0


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

    def build_candidates(
            self, gmrf, sampled_variables, ucb_coefficient,
            threshold, eligible_variables=None):
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError('candidate threshold must be in [0, 1]')
        ucb_values = normalized_ucb(
            gmrf.solution, gmrf.variance, ucb_coefficient)

        candidates = []
        for variable in sorted(self.available_variables(
                len(gmrf.var_cells), sampled_variables,
                eligible_variables)):
            ucb = float(ucb_values[variable])
            if ucb < threshold:
                continue
            row, col = gmrf.var_cells[variable]
            x, y = gmrf.cell_center(variable)
            candidates.append(HrsCandidate(
                variable=variable,
                row=int(row),
                col=int(col),
                x=x,
                y=y,
                ucb=ucb,
                score=ucb,
                mean=float(gmrf.solution[variable]),
                variance=float(gmrf.variance[variable])))

        candidates.sort(key=lambda cell: (
            cell.row, cell.col, cell.variable))
        return tuple(candidates)

    @staticmethod
    def select_candidate(
            candidates, current_xy, distance_fn, distance_weight):
        """Score current candidates and select exactly one DD-UCB target."""
        current = (float(current_xy[0]), float(current_xy[1]))
        if not all(math.isfinite(value) for value in current):
            raise ValueError('current_xy must contain finite coordinates')
        candidates = tuple(candidates)
        if not candidates:
            return (), None
        distances = tuple(float(distance_fn(
            current, (candidate.x, candidate.y)))
            for candidate in candidates)
        scores, normalized_ucb_values, normalized_distances = (
            distance_aware_scores(
                [candidate.score if candidate.ucb is None else candidate.ucb
                 for candidate in candidates], distances,
                distance_weight))
        scored = tuple(
            replace(
                candidate,
                ucb=float(
                    candidate.score if candidate.ucb is None
                    else candidate.ucb),
                score=float(scores[index]),
                distance=float(distances[index]),
                normalized_ucb=float(normalized_ucb_values[index]),
                normalized_distance=float(normalized_distances[index]))
            for index, candidate in enumerate(candidates))
        selected = min(scored, key=lambda candidate: (
            -candidate.score,
            -(candidate.score if candidate.ucb is None else candidate.ucb),
            candidate.row, candidate.col, candidate.variable))
        return scored, selected

    @staticmethod
    def reached_response_threshold(value, threshold):
        value = float(value)
        threshold = float(threshold)
        if not math.isfinite(value):
            raise ValueError('measured value must be finite')
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError('response threshold must be in [0, 1]')
        return value >= threshold


__all__ = ['HrsCandidate', 'HrsManager']
