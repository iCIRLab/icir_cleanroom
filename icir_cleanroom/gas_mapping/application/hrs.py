"""HRS runtime-state owner, candidate policy, and exact planning facade."""

from ..models import HrsRuntimeState
from ..planning.hrs import HrsCell, solve_hrs_p2
from ..planning.hrs_policy import normalized_ucb, peak_search_status


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
            eligible_variables=None):
        potentials = normalized_ucb(
            gmrf.solution, gmrf.variance, float(ucb_coefficient))
        candidates = []
        for variable in self.available_variables(
                len(gmrf.var_cells), sampled_variables,
                eligible_variables):
            row, col = gmrf.var_cells[variable]
            x, y = gmrf.cell_center(variable)
            candidates.append(HrsCell(
                variable=variable, row=int(row), col=int(col), x=x, y=y,
                reward=float(potentials[variable]),
                mean=float(gmrf.solution[variable]),
                variance=float(gmrf.variance[variable])))
        candidates.sort(key=lambda cell: (-cell.reward, cell.row, cell.col))
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
    def stop_decision(cycles, converged, minimum_cycles, maximum_cycles):
        if int(cycles) >= int(minimum_cycles) and bool(converged):
            return 'converged'
        if int(cycles) >= int(maximum_cycles):
            return 'max_cycles'
        return None

    def evaluate_stop(
            self, gmrf, sampled_variables, best_observed,
            ucb_coefficient, margin, eligible_variables=None):
        available = self.available_variables(
            len(gmrf.var_cells), sampled_variables, eligible_variables)
        return peak_search_status(
            gmrf.solution, gmrf.variance, available, best_observed,
            ucb_coefficient, margin)


__all__ = ['HrsManager']
