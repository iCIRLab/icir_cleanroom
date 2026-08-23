"""Hrs Workflow adapter."""

from ..application.hrs import HrsManager
from ..models import PlanningKind


class HrsWorkflow:
    METHODS = ['available_variables','build_candidates','start_hrs_planning','plan_with_fallback','finish_hrs_cycle','hrs_stop_decision','evaluate_hrs_stop']

    def __init__(self, controller):
        self.controller = controller

    def available_variables(self):
        return self.controller.hrs_manager.available_variables(
            len(self.controller.gmrf.var_cells),
            self.controller.sampled_variables,
            self.controller.navigation_goal_variables)

    def build_candidates(self):
        return self.controller.hrs_manager.build_candidates(
            self.controller.gmrf, self.controller.sampled_variables, self.controller.hrs_ucb_k,
            self.controller.hrs_candidate_count,
            self.controller.navigation_goal_variables)

    def start_hrs_planning(self):
        if self.controller.planning_executor.active_task is not None:
            return
        available = self.controller.available_variables()
        if not available:
            reason = ('all reachable cells sampled' if not self.controller.unreachable_variables
                      else 'no reachable unsampled cells remain')
            self.controller.start_peak_confirmation(reason)
            return
        candidates = self.controller.build_candidates()
        self.controller.publish_candidates(candidates)
        current_xy = (
            self.controller.latest_pose.pose.position.x,
            self.controller.latest_pose.pose.position.y)
        max_visits = min(int(self.controller.hrs_visit_count), len(candidates))
        self.controller.publish_phase('HRS_PLANNING')
        self.controller.get_logger().info(
            f'HRS cycle {self.controller.hrs_cycles + 1}: Q_u={len(available)}, '
            f'candidates={len(candidates)}, '
            f'ucb_k={float(self.controller.hrs_ucb_k):.3f}, '
            f'visit_count={max_visits}, planner={str(self.controller.hrs_planner_mode)}, '
            f'top_M='
            f'{[(cell.row, cell.col, round(cell.reward, 6)) for cell in candidates]}')
        self.controller.planning_executor.submit(
            PlanningKind.HRS, self.controller.phase, self.controller.lrs_lap,
            self.controller.current_event_id, None,
            self.controller.plan_with_fallback, candidates, current_xy, max_visits)

    def plan_with_fallback(self, candidates, current_xy, max_visits):
        return self.controller.hrs_manager.plan_with_fallback(
            candidates, current_xy, max_visits, self.controller.config.hrs,
            self.controller.hrs_dwell_seconds)

    def finish_hrs_cycle(self):
        actual_seconds = (
            (self.controller.get_clock().now().nanoseconds - self.controller.hrs_cycle_started_ns)
            * 1.0e-9)
        deadline = float(self.controller.hrs_update_seconds)
        deadline_text = ('DEADLINE MISS' if actual_seconds > deadline
                         else 'on time')
        self.controller.hrs_cycles += 1
        self.controller.hrs_cycles_in_alert += 1
        self.controller.get_logger().info(
            f'=== HRS cycle {self.controller.hrs_cycles} 완료: '
            f'success={self.controller.hrs_batch_successes}/'
            f'{len(self.controller.active_hrs_route.cells)}, '
            f'expected={self.controller.active_hrs_route.expected_seconds:.3f}s, '
            f'actual={actual_seconds:.3f}s, {deadline_text} ===')
        map_updated = self.controller.finalize_hrs_gmrf_batch(
            f'HRS cycle {self.controller.hrs_cycles}')
        self.controller.publish_hrs_status()
        self.controller.persist_history(f'HRS cycle {self.controller.hrs_cycles}')
        self.controller.active_hrs_route = None
        if not map_updated:
            self.controller.get_logger().warning(
                f'HRS 수렴 판정 보류: '
                f'alert_cycle={self.controller.hrs_cycles_in_alert}, '
                'dirty GaBP map could not be recovered')
            if self.controller.available_variables():
                self.controller.start_hrs_planning()
            else:
                self.controller.return_to_lrs(
                    'peak_unconfirmed: GMRF update failed and no '
                    'unvisited cell remains')
            return
        converged, detail = self.controller.evaluate_hrs_stop()
        self.controller.get_logger().info(detail)
        decision = self.controller.hrs_stop_decision(
            self.controller.hrs_cycles_in_alert, converged,
            int(self.controller.hrs_min_cycles_per_alert),
            int(self.controller.hrs_max_cycles_per_alert))
        if decision == 'converged':
            self.controller.start_peak_confirmation(
                'no reachable unvisited cell has a higher concentration '
                'potential')
        elif decision == 'max_cycles':
            self.controller.start_peak_confirmation(
                f'HRS maximum {self.controller.hrs_max_cycles_per_alert} cycles reached '
                'without convergence')
        else:
            self.controller.start_hrs_planning()

    @staticmethod
    @staticmethod
    def hrs_stop_decision(cycles, converged, minimum_cycles, maximum_cycles):
        return HrsManager.stop_decision(
            cycles, converged, minimum_cycles, maximum_cycles)

    def evaluate_hrs_stop(self):
        converged, variable, maximum, gap = self.controller.hrs_manager.evaluate_stop(
            self.controller.gmrf, self.controller.sampled_variables, self.controller.event_best_observed,
            float(self.controller.hrs_ucb_k), float(self.controller.hrs_stop_margin),
            self.controller.navigation_goal_variables)
        if variable is None:
            return True, (
                f'HRS 수렴 판정: alert_cycle={self.controller.hrs_cycles_in_alert}, '
                f'best_observed={self.controller.event_best_observed:.4f}, '
                'available=0, converged=True')
        row, col = self.controller.gmrf.var_cells[variable]
        x, y = self.controller.gmrf.cell_center(variable)
        return converged, (
            f'HRS 수렴 판정: alert_cycle={self.controller.hrs_cycles_in_alert}, '
            f'best_observed={self.controller.event_best_observed:.4f}, '
            f'best_unvisited_potential={maximum:.4f}, gap={gap:.4f}, '
            f'margin={float(self.controller.hrs_stop_margin):.4f}, '
            f'cell=({int(row)},{int(col)}), '
            f'position=({x:.3f},{y:.3f}), converged={converged}')


__all__ = ['HrsWorkflow']
