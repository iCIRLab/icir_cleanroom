"""Single-worker planning execution with stale-result protection."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..models import MappingPhase, PlanningKind, PlanningTask


@dataclass(frozen=True)
class PlanningOutcome:
    task: PlanningTask
    result: object = None
    error: BaseException | None = None


class PlanningExecutor:
    def __init__(self):
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='gas_mapping_planner')
        self._generation = 0
        self._task = None
        self._future = None

    @property
    def active_task(self):
        return self._task

    @property
    def active_future(self):
        return self._future

    def submit(self, kind, phase, lap, event_id, context, function, *args):
        if self._future is not None:
            raise RuntimeError('another route-planning job is already active')
        self._generation += 1
        task = PlanningTask(
            generation=self._generation, kind=PlanningKind(kind),
            phase=MappingPhase(phase), lap=int(lap), event_id=str(event_id),
            context=context)
        self._task = task
        self._future = self._pool.submit(function, *args)
        return task

    def poll(self):
        if self._future is None or not self._future.done():
            return None
        task, future = self._task, self._future
        self._task = None
        self._future = None
        try:
            return PlanningOutcome(task=task, result=future.result())
        except BaseException as error:  # returned to the ROS error boundary
            return PlanningOutcome(task=task, error=error)

    def accepts(self, task, phase, lap, event_id):
        return (
            task.generation == self._generation and
            task.phase == MappingPhase(phase) and
            task.lap == int(lap) and task.event_id == str(event_id))

    def invalidate(self):
        self._generation += 1
        if self._future is not None:
            self._future.cancel()
        self._task = None
        self._future = None

    def shutdown(self):
        self.invalidate()
        self._pool.shutdown(wait=False, cancel_futures=True)


__all__ = ['PlanningExecutor', 'PlanningOutcome']
