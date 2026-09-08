"""Asynchronous LRS planning workflow adapter."""

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
                task, self.controller.phase, self.controller.lrs_lap,
                self.controller.current_event_id):
            self.controller.get_logger().warning(
                f'오래된 {task.kind.value} planning 결과를 무시합니다: '
                f'generation={task.generation}, phase={task.phase.value}, '
                f'lap={task.lap}, event={task.event_id}')
            return
        if task.kind != PlanningKind.LRS:
            self.controller.get_logger().error(
                f'unsupported planning task: {task.kind}')
            return
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


__all__ = ['PlanningWorkflow']
