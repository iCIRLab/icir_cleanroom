"""Peak Workflow adapter."""

import time
from ..planning.hrs import solve_peak_confirmation_route


class PeakWorkflow:
    METHODS = ['start_peak_confirmation','peak_confirmation_cells','plan_peak_confirmation_neighborhood','finish_peak_confirmation_batch','evaluate_peak_confirmation_neighborhood','start_peak_return','finish_peak_return']

    def __init__(self, controller):
        self.controller = controller

    def start_peak_confirmation(self, trigger_reason):
        self.controller.active_hrs_route = None
        if (self.controller.event_best_variable is None or
                self.controller.event_best_variable not in self.controller.hrs_values):
            self.controller.return_to_lrs(
                f'peak_unconfirmed: no measured anchor ({trigger_reason})')
            return
        self.controller.peak_manager.begin(self.controller.event_best_variable, trigger_reason)
        self.controller.clear_hrs_candidates()
        row, col = self.controller.gmrf.var_cells[self.controller.peak_anchor_variable]
        x, y = self.controller.gmrf.cell_center(self.controller.peak_anchor_variable)
        self.controller.publish_phase('HRS_PEAK_CONFIRMATION')
        self.controller.get_logger().info(
            f'HRS 최고점 확인 시작: trigger={trigger_reason}, '
            f'anchor=({int(row)},{int(col)}), '
            f'position=({x:.3f},{y:.3f}), '
            f'value={self.controller.hrs_values[self.controller.peak_anchor_variable]:.4f}')
        self.controller.plan_peak_confirmation_neighborhood()

    def peak_confirmation_cells(self, variables):
        return self.controller.peak_manager.build_cells(
            self.controller.gmrf, variables, self.controller.hrs_ucb_k)

    def plan_peak_confirmation_neighborhood(self):
        if self.controller.peak_anchor_variable is None:
            self.controller.return_to_lrs('peak_unconfirmed: anchor was lost')
            return
        neighbors = self.controller.gmrf.neighbor_variables(
            self.controller.peak_anchor_variable)
        blocked = [variable for variable in neighbors
                   if variable in self.controller.unreachable_variables]
        if blocked:
            cells = [tuple(int(item) for item in self.controller.gmrf.var_cells[var])
                     for var in blocked]
            self.controller.return_to_lrs(
                f'peak_unconfirmed: unreachable neighbor cells={cells}')
            return
        missing = [variable for variable in neighbors
                   if variable not in self.controller.hrs_values]
        if not missing:
            self.controller.evaluate_peak_confirmation_neighborhood(neighbors)
            return

        cells = self.controller.peak_confirmation_cells(missing)
        current_xy = (
            self.controller.latest_pose.pose.position.x,
            self.controller.latest_pose.pose.position.y)
        plan = solve_peak_confirmation_route(
            cells, current_xy, speed=float(self.controller.hrs_speed),
            dwell_seconds=float(self.controller.hrs_dwell_seconds))
        if plan is None:
            self.controller.return_to_lrs(
                'peak_unconfirmed: neighbor route planning failed')
            return
        self.controller.active_hrs_route = plan
        self.controller.current_index = 0
        self.controller.retry = 0
        self.controller.hrs_batch_successes = 0
        self.controller.publish_hrs_route(plan)
        anchor_row, anchor_col = self.controller.gmrf.var_cells[
            self.controller.peak_anchor_variable]
        self.controller.get_logger().info(
            f'HRS 최고점 이웃 확인: move={self.controller.peak_move_count}, '
            f'anchor=({int(anchor_row)},{int(anchor_col)}), '
            f'unmeasured={len(cells)}, length={plan.length:.3f}m, '
            f'route={[(cell.row, cell.col) for cell in plan.cells]}')
        self.controller.send_current_goal()

    def finish_peak_confirmation_batch(self):
        route = self.controller.active_hrs_route
        self.controller.get_logger().info(
            f'HRS 최고점 이웃 측정 완료: success='
            f'{self.controller.hrs_batch_successes}/{len(route.cells)}')
        map_updated = self.controller.finalize_hrs_gmrf_batch(
            f'HRS peak confirmation move {self.controller.peak_move_count}')
        self.controller.publish_hrs_status()
        self.controller.persist_history(
            f'HRS peak confirmation move {self.controller.peak_move_count}')
        self.controller.active_hrs_route = None
        if not map_updated:
            self.controller.return_to_lrs(
                'peak_unconfirmed: GMRF update failed after neighbor '
                'measurements')
            return
        self.controller.plan_peak_confirmation_neighborhood()

    def evaluate_peak_confirmation_neighborhood(self, neighbors):
        anchor = self.controller.peak_anchor_variable
        try:
            higher = self.controller.peak_manager.higher_neighbor(
                anchor, neighbors, self.controller.hrs_values, self.controller.gmrf.var_cells,
                float(self.controller.hrs_peak_improvement_epsilon))
        except ValueError as error:
            self.controller.return_to_lrs(f'peak_unconfirmed: {error}')
            return
        if higher is None:
            self.controller.start_peak_return()
            return
        if self.controller.peak_move_count >= int(self.controller.hrs_peak_max_moves):
            row, col = self.controller.gmrf.var_cells[anchor]
            self.controller.return_to_lrs(
                f'peak_unconfirmed: maximum {self.controller.hrs_peak_max_moves} '
                f'center moves reached at ({int(row)},{int(col)})')
            return
        old_row, old_col = self.controller.gmrf.var_cells[anchor]
        new_row, new_col = self.controller.gmrf.var_cells[higher]
        old_value = float(self.controller.hrs_values[anchor])
        new_value = float(self.controller.hrs_values[higher])
        self.controller.peak_manager.move_anchor(higher)
        self.controller.get_logger().info(
            f'HRS 최고점 중심 이동 {self.controller.peak_move_count}: '
            f'({int(old_row)},{int(old_col)}) {old_value:.4f} -> '
            f'({int(new_row)},{int(new_col)}) {new_value:.4f}')
        self.controller.plan_peak_confirmation_neighborhood()

    def start_peak_return(self):
        anchor = self.controller.peak_anchor_variable
        cell = self.controller.peak_confirmation_cells([anchor])[0]
        current_xy = (
            self.controller.latest_pose.pose.position.x,
            self.controller.latest_pose.pose.position.y)
        plan = solve_peak_confirmation_route(
            [cell], current_xy, speed=float(self.controller.hrs_speed),
            dwell_seconds=0.0)
        if plan is None:
            self.controller.return_to_lrs(
                'peak_unconfirmed: confirmed-peak return planning failed')
            return
        self.controller.peak_returning = True
        self.controller.active_hrs_route = plan
        self.controller.current_index = 0
        self.controller.retry = 0
        self.controller.publish_hrs_route(plan)
        self.controller.get_logger().info(
            f'HRS 국소 최고 셀 복귀: cell=({cell.row},{cell.col}), '
            f'position=({cell.x:.3f},{cell.y:.3f}), '
            f'value={self.controller.hrs_values[anchor]:.4f}, '
            f'distance={plan.length:.3f}m')
        self.controller.send_current_goal()

    def finish_peak_return(self):
        anchor = self.controller.peak_anchor_variable
        row, col = self.controller.gmrf.var_cells[anchor]
        x, y = self.controller.gmrf.cell_center(anchor)
        value = float(self.controller.hrs_values[anchor])
        actual_x = self.controller.latest_pose.pose.position.x
        actual_y = self.controller.latest_pose.pose.position.y
        self.controller.active_hrs_route = None
        self.controller.peak_returning = False
        actual_variable = self.controller.gmrf.nearest_variable(actual_x, actual_y)
        if actual_variable != anchor:
            actual_row, actual_col = self.controller.gmrf.var_cells[actual_variable]
            self.controller.return_to_lrs(
                f'peak_unconfirmed: peak return reached '
                f'({int(actual_row)},{int(actual_col)}) instead of '
                f'({int(row)},{int(col)})')
            return
        self.controller.get_logger().info(
            f'HRS 실측 국소 최고점 도착 및 확정: '
            f'cell=({int(row)},{int(col)}), '
            f'target=({x:.3f},{y:.3f}), '
            f'robot=({actual_x:.3f},{actual_y:.3f}), value={value:.4f}, '
            f'moves={self.controller.peak_move_count}, '
            f'trigger={self.controller.peak_trigger_reason}')
        try:
            inserted = self.controller.history.record_confirmed_peak(
                self.controller.current_event_id, int(row), int(col), value,
                time.time())
        except ValueError as error:
            self.controller.get_logger().error(
                f'확정 최고점 이벤트 기록 실패: {error}')
            inserted = False
        if inserted:
            self.controller.get_logger().info(
                f'독립 가스 이벤트 확정 기록: id={self.controller.current_event_id}, '
                f'cell=({int(row)},{int(col)}), value={value:.4f}')
        else:
            self.controller.get_logger().warning(
                f'이미 기록된 가스 이벤트를 중복 집계하지 않습니다: '
                f'id={self.controller.current_event_id}')
        self.controller.publish_history()
        self.controller.start_source_transition(
            f'peak_confirmed at ({int(row)},{int(col)}) value={value:.4f}')


__all__ = ['PeakWorkflow']

