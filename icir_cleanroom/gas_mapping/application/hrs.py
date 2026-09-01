"""HRS runtime-state owner, candidate policy, and exact planning facade."""
import math

from ..models import HrsRuntimeState
from ..planning.hrs import HrsCell, solve_hrs_p2
from ..planning.hrs_policy import distance_discounted_ucb


class HrsManager:
    def __init__(self, state=None):
        self.state = state or HrsRuntimeState()

    def reset_search(self):
        self.state.reset_search()
        self.state.active_route = None
        self.state.cycle_started_ns = None

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
        self.state.batch_successes += 1

    def build_candidates(
            self, gmrf, sampled_variables, ucb_coefficient, count,
            eligible_variables=None, current_xy=None, distance_weight=0.0):
        distance_weight = float(distance_weight)
        if not math.isfinite(distance_weight) or distance_weight < 0.0:
            raise ValueError(
                'distance_weight must be finite and non-negative')

        if distance_weight > 0.0 and current_xy is None:
            raise ValueError(
                'current_xy is required when distance_weight is positive')

        if current_xy is not None:
            current_x, current_y = (
                float(current_xy[0]), float(current_xy[1]))
            if not math.isfinite(current_x) or not math.isfinite(current_y):
                raise ValueError('current_xy must contain finite coordinates')
        else:
            current_x = current_y = 0.0

        distances = []
        for variable in range(len(gmrf.var_cells)):
            x, y = gmrf.cell_center(variable)
            distances.append(
                math.hypot(x - current_x, y - current_y)
                if current_xy is not None else 0.0)
        potentials = distance_discounted_ucb(
            gmrf.solution, gmrf.variance, distances,
            float(ucb_coefficient), distance_weight)

        candidates = []
        for variable in self.available_variables(
                len(gmrf.var_cells), sampled_variables,
                eligible_variables):
            row, col = gmrf.var_cells[variable]
            x, y = gmrf.cell_center(variable)

            reward = float(potentials[variable])

            candidates.append(HrsCell(
                variable=variable,
                row=int(row),
                col=int(col),
                x=x,
                y=y,
                reward=reward,
                mean=float(gmrf.solution[variable]),
                variance=float(gmrf.variance[variable])))

        candidates.sort(
            key=lambda cell: (-cell.reward, cell.row, cell.col))
        return candidates[:int(count)]

    @staticmethod
    def plan_with_fallback(
            candidates, current_xy, max_visits, hrs_config,
            dwell_seconds):
        attempts = []
        for visit_count in range(int(max_visits), 0, -1):
            plan = solve_hrs_p2(
                candidates, current_xy, visit_count=visit_count,
                speed=float(hrs_config.hrs_speed),
                update_seconds=float(hrs_config.hrs_update_seconds),
                dwell_seconds=float(dwell_seconds),
                combination_time_limit=float(
                    hrs_config.hrs_combination_time_limit),
                planner_mode=str(hrs_config.hrs_planner_mode))
            attempts.append((visit_count, plan))
            if plan is not None:
                return plan, attempts
        return None, attempts

    @staticmethod
    def reached_response_threshold(value, threshold):
        value = float(value)
        threshold = float(threshold)
        if not math.isfinite(value):
            raise ValueError('measured value must be finite')
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError('response threshold must be in [0, 1]')
        return value >= threshold


__all__ = ['HrsManager']
