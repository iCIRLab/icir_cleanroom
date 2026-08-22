"""Characterization tests for extracted application services."""

from types import SimpleNamespace

import numpy as np

from icir_cleanroom.gas_mapping.application.measurement import (
    MeasurementManager)
from icir_cleanroom.gas_mapping.application.navigation import NavigationManager
from icir_cleanroom.gas_mapping.application.planning_executor import (
    PlanningExecutor)
from icir_cleanroom.gas_mapping.config import ControllerConfig
from icir_cleanroom.gas_mapping.mapping.field_projection import (
    logarithmic_display, project_field)
from icir_cleanroom.gas_mapping.mapping.gmrf import GmrfGrid
from icir_cleanroom.gas_mapping.mapping.gmrf_service import GmrfService
from icir_cleanroom.gas_mapping.models import MappingPhase, PlanningKind


def test_gmrf_service_rolls_back_every_mutable_array(map_message_factory):
    gmrf = GmrfGrid(map_message_factory())
    gmrf.solution[:] = np.arange(9) / 10.0
    gmrf.variance[:] = 0.25
    gmrf.message_precision[:] = 0.2
    gmrf.message_information[:] = 0.3
    snapshots = [array.copy() for array in (
        gmrf.solution, gmrf.variance, gmrf.message_precision,
        gmrf.message_information)]

    def fail(*_args):
        gmrf.solution.fill(1.0)
        gmrf.variance.fill(2.0)
        gmrf.message_precision.fill(3.0)
        gmrf.message_information.fill(4.0)
        return False, 7, 0.5

    gmrf.solve_gabp = fail
    service = GmrfService(ControllerConfig.defaults().gmrf, gmrf)
    result = service.update()

    assert not result.success
    for actual, expected in zip((
            gmrf.solution, gmrf.variance, gmrf.message_precision,
            gmrf.message_information), snapshots):
        np.testing.assert_array_equal(actual, expected)


def test_projection_and_log_transform_are_ros_independent(map_message_factory):
    gmrf = GmrfGrid(map_message_factory())
    projected = project_field(gmrf, np.arange(9) / 10.0, 0.5)
    assert (projected.width, projected.height) == (5, 5)
    assert projected.values.shape == (5, 5)
    assert projected.free_mask.all()
    displayed = logarithmic_display(np.asarray([0.0, 1.0]), -4.0)
    np.testing.assert_allclose(displayed, [0.0, 1.0])


def test_measurement_manager_rejects_stale_dwell_and_commits_one_record():
    manager = MeasurementManager()
    pose = SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=2.0, y=3.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))
    generation = manager.start(pose, MappingPhase.LRS)
    assert manager.add_sample(0.2)
    assert manager.add_sample(0.4)
    assert manager.finish(generation - 1) is None
    result = manager.finish(generation)
    measurement = manager.commit(result, variable=5, timestamp=10.0)

    assert result.mean == 0.30000000000000004
    assert measurement.variable == 5
    assert measurement.sample_count == 2
    assert len(manager.measurements) == 1


def test_navigation_manager_invalidates_old_goal_callbacks():
    manager = NavigationManager()
    first = manager.issue_goal()
    second = manager.issue_goal()
    assert not manager.accepts(first)
    assert manager.accepts(second)


def test_planning_executor_tags_and_returns_the_active_task():
    executor = PlanningExecutor()
    try:
        task = executor.submit(
            PlanningKind.LRS, MappingPhase.LRS_PLANNING, 2, 'event-1',
            {'input': 3}, lambda value: value + 1, 3)
        outcome = None
        while outcome is None:
            outcome = executor.poll()
        assert outcome.error is None
        assert outcome.result == 4
        assert outcome.task == task
        assert executor.accepts(
            task, MappingPhase.LRS_PLANNING, 2, 'event-1')
        assert not executor.accepts(
            task, MappingPhase.HRS_PLANNING, 2, 'event-1')
    finally:
        executor.shutdown()
