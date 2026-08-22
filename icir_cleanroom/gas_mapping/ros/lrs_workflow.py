"""Lrs Workflow adapter."""

from std_msgs.msg import Bool, Int32
from ..application.lrs import LrsManager
from ..models import PlanningKind
from ..planning.lrs_priority import (
    history_adjusted_cycle, solve_lrs_priority_route)


class LrsWorkflow:
    METHODS = ['start_lrs_lap','prepare_adaptive_lrs_points','finish_lrs_route_planning','closed_cycle_reference_length','finish_lrs_navigation']

    def __init__(self, controller):
        self.controller = controller

    def start_lrs_lap(self, reason):
        if not self.controller.base_lrs_goals or self.controller.latest_pose is None:
            return
        self.controller.orchestrator.begin_lrs_lap()
        self.controller.publish_hrs_status()

        self.controller.gmrf.reset_observations()
        if not self.controller.update_gmrf(f'LRS lap {self.controller.lrs_lap} reset'):
            self.controller.get_logger().warning(
                '새 LRS 회차의 background-prior GMRF 초기화가 '
                '수렴하지 않았습니다')

        self.controller.active_history_regions = self.controller.history.confirmed_regions(
            top_k=int(self.controller.history_top_k),
            merge_radius=float(self.controller.history_merge_radius),
            event_half_life=float(self.controller.history_event_half_life))
        base_points = [
            (goal.pose.position.x, goal.pose.position.y)
            for goal in self.controller.base_lrs_goals]
        history_points = [
            (record['x'], record['y'])
            for record in self.controller.active_history_regions]
        current_xy = (
            self.controller.latest_pose.pose.position.x,
            self.controller.latest_pose.pose.position.y)
        adjusted, replacements, additions = history_adjusted_cycle(
            base_points, history_points,
            float(self.controller.lrs_history_replace_radius), current_xy)
        points = self.controller.prepare_adaptive_lrs_points(adjusted)
        if self.controller.planning_executor.active_task is not None:
            raise RuntimeError('another route-planning job is already active')
        planning_context = {
            'reason': str(reason), 'base_points': base_points,
            'history_points': history_points,
            'replacements': replacements, 'additions': additions,
            'points': points, 'current_xy': current_xy,
            'original_baseline_length': self.controller.closed_cycle_reference_length(
                base_points, current_xy),
        }
        self.controller.lrs_goals = []
        self.controller.lrs_status_values = []
        self.controller.lrs_priority_points = []
        self.controller.publish_phase('LRS_PLANNING')
        self.controller.hazard_pub.publish(Bool(data=False))
        self.controller.lrs_lap_pub.publish(Int32(data=self.controller.lrs_lap))
        self.controller.publish_lrs_active_route()
        self.controller.publish_lrs_reward()
        self.controller.publish_lrs_priority_candidates()
        self.controller.publish_lrs_priority_route()
        self.controller.publish_lrs_status()
        self.controller.publish_history()
        self.controller.get_logger().info(
            f'=== LRS lap {self.controller.lrs_lap} 경로 계산 시작 ({reason}): '
            f'base={len(base_points)}, active={len(points)}, '
            f'history_regions={len(history_points)}, '
            f'replaced={len(replacements)}, added={len(additions)} ===')
        if replacements:
            self.controller.get_logger().info(
                'LRS 역사 위치 대체: ' + str([
                    (tuple(round(value, 3) for value in old),
                     tuple(round(value, 3) for value in new),
                     round(separation, 3))
                    for old, new, separation in replacements]))
        if additions:
            self.controller.get_logger().info(
                'LRS 역사 위치 추가: ' + str([
                    tuple(round(value, 3) for value in point)
                    for point in additions]))
        self.controller.planning_executor.submit(
            PlanningKind.LRS, self.controller.phase, self.controller.lrs_lap,
            self.controller.current_event_id, planning_context,
            solve_lrs_priority_route, points, current_xy,
            int(self.controller.lrs_priority_candidate_count),
            int(self.controller.lrs_priority_count),
            float(self.controller.lrs_route_length_ratio))

    def prepare_adaptive_lrs_points(self, adjusted_points):
        """Calculate active-set LRS rewards before threaded route solving."""
        points, duplicates = self.controller.lrs_manager.prepare_reward_points(
            adjusted_points, self.controller.gmrf, self.controller.history, self.controller.config.lrs,
            self.controller.config.history)
        for variable in duplicates:
            row, col = self.controller.gmrf.var_cells[variable]
            self.controller.get_logger().warning(
                f'중복 LRS GMRF 셀 ({int(row)},{int(col)})을 '
                '이번 활성 경로에서 한 번만 유지합니다')
        for point in self.controller.lrs_priority_candidates:
            self.controller.get_logger().info(
                f'LRS reward cell=({point.row},{point.col}), '
                f'total={point.reward:.4f}, '
                f'recurrence={point.recurrence:.4f}, '
                f'severity={point.severity:.4f}, '
                f'staleness={point.staleness:.4f}, '
                f'uncertainty={point.uncertainty:.4f}, '
                f'mandatory={point.mandatory}')
        return points

    def finish_lrs_route_planning(self, route_plan, context):
        if self.controller.phase != 'LRS_PLANNING':
            self.controller.get_logger().warning(
                f'현재 phase={self.controller.phase}이므로 완료된 LRS 경로를 '
                '적용하지 않습니다')
            return
        self.controller.lrs_reward_points = list(route_plan.points)
        self.controller.lrs_priority_points = list(route_plan.priority_points)
        self.controller.lrs_goals = [
            self.controller.make_pose(point.x, point.y) for point in route_plan.points]
        self.controller.lrs_status_values = [None] * len(self.controller.lrs_goals)
        self.controller.current_index = 0
        self.controller.retry = 0
        self.controller.returning = False

        self.controller.publish_phase('LRS')
        self.controller.publish_lrs_active_route()
        self.controller.publish_lrs_priority_route()
        self.controller.publish_lrs_status()
        base_points = context['base_points']
        history_points = context['history_points']
        replacements = context['replacements']
        additions = context['additions']
        self.controller.get_logger().info(
            f'=== LRS lap {self.controller.lrs_lap} 시작 ({context["reason"]}): '
            f'base={len(base_points)}, active={len(self.controller.lrs_goals)}, '
            f'history_regions={len(history_points)}, '
            f'replaced={len(replacements)}, added={len(additions)}, '
            f'priority={route_plan.priority_count}, '
            f'route={route_plan.length:.3f}m, '
            f'active_baseline={route_plan.baseline_length:.3f}m, '
            f'original_p1={context["original_baseline_length"]:.3f}m, '
            f'({route_plan.length_ratio:.3f}x), '
            f'fallback={route_plan.fallback} ===')
        self.controller.get_logger().info(
            f'LRS reward 경로 계산: status={route_plan.status}, '
            f'candidates={len(self.controller.lrs_priority_candidates)}, '
            f'total_combinations={route_plan.total_combinations}, '
            f'routes_solved={route_plan.evaluated_routes}, '
            f'feasible={route_plan.feasible_routes}, '
            f'priority_reward={route_plan.reward:.6f}, '
            f'planning={route_plan.planning_seconds:.4f}s, '
            f'priority_cells='
            f'{[(point.row, point.col) for point in route_plan.priority_points]}')
        self.controller.send_current_goal()

    @staticmethod
    @staticmethod
    def closed_cycle_reference_length(points, start_xy):
        return LrsManager.closed_cycle_reference_length(points, start_xy)

    def finish_lrs_navigation(self):
        self.controller.returning = False
        self.controller.get_logger().info(
            f'=== LRS lap {self.controller.lrs_lap} 완료: '
            f'max={self.controller.lap_max_concentration:.4f}, '
            f'threshold={float(self.controller.hazard_threshold):.4f}, '
            f'hazard={self.controller.lap_hazard_detected} ===')
        if self.controller.lap_hazard_detected:
            self.controller.persist_history('hazardous LRS measurements')
            self.controller.hrs_cycles_in_alert = 0
            self.controller.start_hrs_planning()
        else:
            self.controller.commit_history_snapshot(
                f'non-hazardous LRS lap {self.controller.lrs_lap}')
            self.controller.start_lrs_lap('no hazardous concentration detected')


__all__ = ['LrsWorkflow']
