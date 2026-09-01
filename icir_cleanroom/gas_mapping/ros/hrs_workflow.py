"""Hrs Workflow adapter."""

from ..models import PlanningKind


class HrsWorkflow:
    METHODS = [
        'available_variables', 'build_candidates', 'start_hrs_planning',
        'plan_with_fallback', 'confirm_hrs_response', 'finish_hrs_cycle']

    def __init__(self, controller):
        self.controller = controller

    def available_variables(self):
        return self.controller.hrs_manager.available_variables(
            len(self.controller.gmrf.var_cells),
            self.controller.sampled_variables,
            self.controller.navigation_goal_variables)

    def build_candidates(self, current_xy):
        return self.controller.hrs_manager.build_candidates(
            self.controller.gmrf, self.controller.sampled_variables, self.controller.hrs_ucb_k,
            self.controller.hrs_candidate_count,
            self.controller.navigation_goal_variables,
            current_xy=current_xy,
            distance_weight=self.controller.hrs_distance_weight)

    def start_hrs_planning(self):
        if self.controller.planning_executor.active_task is not None:
            return
        available = self.controller.available_variables()
        if not available:
            reason = ('all reachable cells sampled' if not self.controller.unreachable_variables
                      else 'no reachable unsampled cells remain')
            self.controller.return_to_lrs(
                f'HRS response threshold not reached: {reason}')
            return
        current_xy = (
            self.controller.latest_pose.pose.position.x,
            self.controller.latest_pose.pose.position.y)
        self.controller.publish_dd_ucb(current_xy)
        candidates = self.controller.build_candidates(current_xy)
        self.controller.publish_candidates(candidates)
        max_visits = min(int(self.controller.hrs_visit_count), len(candidates))
        self.controller.publish_phase('HRS_PLANNING')
        self.controller.get_logger().info(
            f'HRS cycle {self.controller.hrs_cycles + 1}: Q_u={len(available)}, '
            f'candidates={len(candidates)}, '
            f'ucb_k={float(self.controller.hrs_ucb_k):.3f}, '
            f'distance_weight={float(self.controller.hrs_distance_weight):.3f}, '
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

    def confirm_hrs_response(self, variable, value, timestamp):
        threshold = float(self.controller.hrs_response_threshold)
        if not self.controller.hrs_manager.reached_response_threshold(
                value, threshold):
            return False

        row, col = self.controller.gmrf.var_cells[int(variable)]
        try:
            inserted = self.controller.history.record_confirmed_event(
                self.controller.current_event_id, int(row), int(col),
                float(value), threshold, float(timestamp),
                method='threshold_crossing')
        except (TypeError, ValueError) as error:
            self.controller.get_logger().error(
                f'HRS 확정 검출 이벤트 기록 실패: {error}')
            inserted = False
        if inserted:
            self.controller.get_logger().warning(
                f'HRS 대응 임계값 검출: cell=({int(row)},{int(col)}), '
                f'value={float(value):.4f} >= threshold={threshold:.4f}')
        else:
            self.controller.get_logger().warning(
                f'HRS 대응 이벤트가 이미 기록되었거나 저장할 수 없습니다: '
                f'event={self.controller.current_event_id}')

        # Persist the authoritative measurement event before any optional
        # GMRF recovery or external source-transition operation.
        self.controller.publish_history()
        self.controller.persist_history('HRS response threshold detection')
        if self.controller.hrs_gmrf_dirty:
            if not self.controller.finalize_hrs_gmrf_batch(
                    'HRS response threshold detection'):
                self.controller.get_logger().warning(
                    '확정 검출은 유지하지만 최종 GMRF 복구에는 실패했습니다')
        self.controller.start_source_transition(
            f'response threshold detected at ({int(row)},{int(col)}), '
            f'value={float(value):.4f}, threshold={threshold:.4f}')
        return True

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
            self.controller.return_to_lrs(
                'HRS response threshold not reached: GMRF update failed')
            return
        if (self.controller.hrs_cycles_in_alert >=
                int(self.controller.hrs_max_cycles_per_alert)):
            self.controller.return_to_lrs(
                f'HRS response threshold {float(self.controller.hrs_response_threshold):.4f} '
                f'not reached after {self.controller.hrs_cycles_in_alert} cycles')
            return
        if not self.controller.available_variables():
            self.controller.return_to_lrs(
                'HRS response threshold not reached: no reachable '
                'unvisited cells remain')
            return
        self.controller.start_hrs_planning()


__all__ = ['HrsWorkflow']
