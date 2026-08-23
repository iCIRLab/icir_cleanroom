"""Planning Workflow adapter."""

from ..models import PlanningKind
from ..planning.lrs_priority import lrs_coverage_fallback


class PlanningWorkflow:
    METHODS = ['poll_planning']

    def __init__(self, controller):
        self.controller = controller

    def poll_planning(self):
        outcome = self.controller.planning_executor.poll()
        if outcome is None:
            return
        task = outcome.task
        if not self.controller.planning_executor.accepts(
                task, self.controller.phase, self.controller.lrs_lap, self.controller.current_event_id):
            self.controller.get_logger().warning(
                f'오래된 {task.kind.value} planning 결과를 무시합니다: '
                f'generation={task.generation}, phase={task.phase.value}, '
                f'lap={task.lap}, event={task.event_id}')
            return
        if task.kind == PlanningKind.LRS:
            context = task.context
            if context is None:
                self.controller.get_logger().error(
                    'LRS planning context가 없어 완료 결과를 버립니다')
                return
            if outcome.error is not None:
                self.controller.get_logger().error(
                    f'LRS reward route worker failed: {outcome.error!r}; '
                    '최단 coverage fallback을 사용합니다')
                route_plan = lrs_coverage_fallback(
                    context['points'], context['current_xy'],
                    distance_fn=context['distance_fn'])
            else:
                route_plan = outcome.result
            self.controller.finish_lrs_route_planning(route_plan, context)
            return
        if outcome.error is not None:
            self.controller.get_logger().error(
                f'HRS P2 worker failed: {outcome.error!r}')
            self.controller.start_peak_confirmation('P2 worker failure')
            return
        plan, attempts = outcome.result
        for visit_count, result in attempts:
            if result is None:
                self.controller.get_logger().warning(
                    f'HRS P2 T={visit_count}: feasible route 없음')
        if plan is None:
            self.controller.start_peak_confirmation('no feasible P2 route')
            return
        self.controller.active_hrs_route = plan
        self.controller.current_index = 0
        self.controller.retry = 0
        self.controller.hrs_batch_successes = 0
        self.controller.hrs_cycle_started_ns = self.controller.get_clock().now().nanoseconds
        self.controller.publish_hrs_route(plan)
        self.controller.publish_phase('HRS_NAVIGATION')
        self.controller.get_logger().info(
            f'HRS P2 선택: T={plan.visit_count}, '
            f'mode={str(self.controller.hrs_planner_mode)}, solver={plan.solver}, '
            f'total_combinations={plan.total_combinations}, '
            f'routes_solved={plan.evaluated_routes}, '
            f'lower_bound_pruned={plan.pruned_combinations}, '
            f'feasible={plan.feasible_combinations}, reward={plan.reward:.6f}, '
            f'length={plan.length:.3f}m, expected={plan.expected_seconds:.3f}s, '
            f'planning={plan.planning_seconds:.3f}s, '
            f'status={plan.status}, gap={plan.gap:.6f}, cuts={plan.cuts}, '
            f'selected={[(cell.row, cell.col) for cell in plan.cells]}')
        self.controller.send_current_goal()


__all__ = ['PlanningWorkflow']
