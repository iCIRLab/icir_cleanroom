"""Tests for typed configuration and explicit runtime state."""

import math
from types import SimpleNamespace

import pytest

from icir_cleanroom.gas_mapping.config import (
    ControllerConfig, DEFAULT_CONTROLLER_PARAMETERS)
from icir_cleanroom.gas_mapping.application.hrs import (
    HrsCandidate, HrsManager)
from icir_cleanroom.gas_mapping.application.lrs import LrsManager
from icir_cleanroom.gas_mapping.application.navigation import NavigationManager
from icir_cleanroom.gas_mapping.application.orchestrator import (
    MappingOrchestrator)
from icir_cleanroom.gas_mapping.models import (
    HrsRuntimeState, MappingPhase, NavigationState)
from icir_cleanroom.gas_mapping.phase_machine import (
    InvalidPhaseTransition, PhaseMachine)
from icir_cleanroom.gas_mapping.ros.hrs_workflow import HrsWorkflow
from icir_cleanroom.gas_mapping.ros.navigation_workflow import (
    NavigationWorkflow)


def controller_values(**overrides):
    values = dict(DEFAULT_CONTROLLER_PARAMETERS)
    values.update(overrides)
    return values


def test_typed_config_round_trips_without_required_parameters():
    values = controller_values()
    config = ControllerConfig.from_mapping(values)
    assert config.flat_values() == values
    assert config.hrs.hrs_distance_weight == 0.03
    assert config.hrs.hrs_response_threshold == 0.9
    assert config.lrs.lrs_priority_count == 6
    assert config.gmrf.gabp_damping == 0.5


def test_config_rejects_invalid_cross_field_values():
    values = controller_values()
    values['lrs_priority_count'] = values['lrs_priority_candidate_count'] + 1
    with pytest.raises(ValueError):
        ControllerConfig.from_mapping(values)

    values = controller_values()
    values['hrs_response_threshold'] = 1.01
    with pytest.raises(ValueError):
        ControllerConfig.from_mapping(values)

    ControllerConfig.from_mapping(DEFAULT_CONTROLLER_PARAMETERS)

    values = controller_values(hrs_distance_weight=-0.01)
    with pytest.raises(ValueError, match='hrs_distance_weight'):
        ControllerConfig.from_mapping(values)


def test_phase_machine_guards_transitions_and_stale_generations():
    phases = PhaseMachine()
    assert phases.accepts(MappingPhase.WAITING, 0)
    phases.transition(MappingPhase.LRS_PLANNING)
    generation = phases.generation
    assert phases.accepts(MappingPhase.LRS_PLANNING, generation)
    assert not phases.accepts(MappingPhase.LRS_PLANNING, generation - 1)
    with pytest.raises(InvalidPhaseTransition):
        phases.transition(MappingPhase.SOURCE_TRANSITION)


def test_phase_machine_allows_deferred_hrs_source_transition():
    phases = PhaseMachine(MappingPhase.HRS_PLANNING)

    phases.transition(MappingPhase.SOURCE_TRANSITION)

    assert phases.accepts(MappingPhase.SOURCE_TRANSITION, phases.generation)


def test_runtime_reset_methods_advance_or_clear_owned_state():
    navigation = NavigationState(index=4, retry_count=2, returning=True)
    navigation.reset_progress()
    assert (navigation.generation, navigation.index,
            navigation.retry_count, navigation.returning) == (1, 0, 0, False)

    hrs = HrsRuntimeState(
        cycles=3, active_target=object(), failure_counts={7: 2},
        route_targets=[object()], active_region_cells={(1, 2)},
        completed_region_cells={(3, 4)}, region_ascent_floor=0.4,
        unreachable_variables={7}, candidate_variables={1, 7}, dirty=True,
        confirmed_variable=2, confirmed_value=0.9,
        confirmed_timestamp=12.0)
    hrs.reset_search()
    assert hrs.cycles == 0
    assert hrs.active_target is None
    assert hrs.route_targets == []
    assert hrs.active_region_cells == set()
    assert hrs.completed_region_cells == set()
    assert hrs.region_ascent_floor is None
    assert hrs.failure_counts == {}
    assert hrs.unreachable_variables == set()
    assert hrs.candidate_variables == set()
    assert hrs.dirty is False
    assert hrs.confirmed_variable is None
    assert hrs.confirmed_value is None
    assert hrs.confirmed_timestamp is None


def test_orchestrator_tracks_sampled_event_measurements():
    phases = PhaseMachine()
    event_orchestrator = MappingOrchestrator(
        phases, NavigationManager(), LrsManager(), HrsManager())
    event_orchestrator.record_event_measurement(0, 0.6)
    value = event_orchestrator.record_event_measurement(2, 0.7)

    assert value == 0.7
    assert event_orchestrator.event.sampled_variables == {0, 2}
    assert event_orchestrator.event.measured_by_variable == {0: 0.6, 2: 0.7}


@pytest.mark.parametrize(
    ('value', 'expected'), [(0.4999, False), (0.5, True), (0.8, True)])
def test_hrs_response_threshold_is_inclusive(value, expected):
    assert HrsManager.reached_response_threshold(value, 0.5) is expected


@pytest.mark.parametrize(
    ('completed_cycles', 'repeat', 'expected_action'),
    [(0, True, 'replan'), (9, True, 'return'),
     (0, False, 'replan'), (9, False, 'complete')])
def test_adaptive_hrs_cycle_replans_until_maximum(
        completed_cycles, repeat, expected_action):
    events = []
    controller = SimpleNamespace(
        repeat_after_hrs=repeat,
        active_hrs_target=SimpleNamespace(variable=3),
        hrs_route_targets=[SimpleNamespace(variable=3)],
        hrs_cycle_started_ns=0,
        hrs_cycles=completed_cycles,
        hrs_cycles_in_alert=completed_cycles,
        hrs_max_cycles_per_alert=10,
        hrs_response_threshold=0.5,
        hrs_confirmed_variable=None,
        hrs_confirmed_value=None,
        hrs_confirmed_timestamp=None,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        get_logger=lambda: SimpleNamespace(
            info=lambda message: None, warning=lambda message: None),
        finalize_hrs_gmrf_batch=lambda reason: events.append('finalize') or True,
        publish_hrs_status=lambda: events.append('publish'),
        persist_history=lambda reason: events.append('persist'),
        start_hrs_planning=lambda: events.append('replan'),
        return_to_lrs=lambda reason: events.append('return'),
        complete_mapping=lambda reason: events.append('complete'))

    HrsWorkflow(controller).finish_hrs_cycle()

    assert controller.active_hrs_target is None
    expected = ['finalize', 'publish', 'persist']
    expected.append(expected_action)
    assert events == expected


@pytest.mark.parametrize('repeat', [True, False])
def test_goal_response_terminates_hrs_immediately(repeat):
    events = []
    history = SimpleNamespace(record_confirmed_event=lambda *args, **kwargs:
                              events.append(('record', args, kwargs)) or True)
    controller = SimpleNamespace(
        repeat_after_hrs=repeat,
        hrs_response_threshold=0.5,
        hrs_manager=HrsManager(),
        gmrf=SimpleNamespace(var_cells=[(2, 3)]),
        history=history,
        current_event_id='event-1',
        hrs_confirmed_variable=None,
        hrs_confirmed_value=None,
        hrs_confirmed_timestamp=None,
        get_logger=lambda: SimpleNamespace(
            warning=lambda message: events.append('warning'),
            error=lambda message: events.append('error'),
            info=lambda message: events.append('info')),
        publish_history=lambda: events.append('publish'),
        persist_history=lambda reason: events.append('persist') or True,
        start_source_transition=lambda reason: events.append('transition'),
        complete_mapping=lambda reason: events.append('complete'))

    workflow = HrsWorkflow(controller)
    assert not workflow.confirm_hrs_response(0, 0.4999, 10.0)
    assert workflow.confirm_hrs_response(0, 0.5, 11.0)
    assert controller.hrs_confirmed_variable == 0
    assert controller.hrs_confirmed_value == 0.5

    record = next(event for event in events
                  if isinstance(event, tuple) and event[0] == 'record')
    assert record[1][0:6] == ('event-1', 2, 3, 0.5, 0.5, 11.0)
    assert events[-4:] == ['publish', 'persist', 'warning',
                           'transition' if repeat else 'complete']


def test_hrs_navigation_never_visits_a_frozen_second_target():
    events = []
    first = SimpleNamespace(variable=1)
    second = SimpleNamespace(variable=2)
    controller = SimpleNamespace(
        repeat_after_hrs=True,
        phase=MappingPhase.HRS_NAVIGATION,
        current_index=0,
        retry=3,
        active_hrs_target=first,
        hrs_route_targets=[first, second],
        publish_hrs_status=lambda: events.append('publish'),
        send_current_goal=lambda: events.append('send'),
        finish_hrs_cycle=lambda: events.append('finish'))
    workflow = NavigationWorkflow(controller)

    workflow.advance_after_target()
    assert controller.current_index == 1
    assert controller.active_hrs_target is first
    assert events == ['publish', 'finish']


def test_subgoal_hrs_dwell_updates_gmrf_and_continues():
    events = []
    pose = SimpleNamespace(position=SimpleNamespace(x=0.5, y=0.5))
    result = SimpleNamespace(
        phase=MappingPhase.HRS_NAVIGATION, mean=0.4999,
        pose=SimpleNamespace(pose=pose), sample_count=3)
    measurement = SimpleNamespace(timestamp=11.0)
    measurement_manager = SimpleNamespace(
        finish=lambda generation: result,
        commit=lambda completed, variable, timestamp: measurement)
    gmrf = SimpleNamespace(
        set_observation=lambda x, y, measured: (0, 0),
        variable_at=lambda row, col: 0)
    history = SimpleNamespace(
        update=lambda *args: events.append('history'))
    target = SimpleNamespace(
        variable=0, row=0, col=0, mean=0.42, score=0.58)
    controller = SimpleNamespace(
        repeat_after_hrs=True,
        phase=MappingPhase.HRS_NAVIGATION,
        measurement_manager=measurement_manager,
        dwell_timer=None,
        gmrf=gmrf,
        history=history,
        active_hrs_target=target,
        hrs_route_targets=[target],
        hrs_cycle_started_ns=0,
        hrs_cycles=0,
        hrs_cycles_in_alert=0,
        hrs_max_cycles_per_alert=10,
        current_index=0,
        hrs_response_threshold=0.5,
        hrs_manager=SimpleNamespace(
            record_success=lambda variable: events.append('success')),
        hrs_gmrf_dirty=False,
        get_logger=lambda: SimpleNamespace(
            warning=lambda message: events.append('warning'),
            info=lambda message: events.append('info')),
        publish_history=lambda: events.append('publish_history'),
        record_event_measurement=lambda variable, measured:
            events.append('record_measurement'),
        update_gmrf=lambda reason, compare_with_cg=False:
            events.append('gmrf_update') or True,
        publish_measurements=lambda: events.append('publish_measurements'),
        confirm_hrs_response=lambda variable, measured, timestamp:
            False,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        finalize_hrs_gmrf_batch=lambda reason:
            events.append('gmrf_finalize') or True,
        publish_hrs_status=lambda: events.append('status'),
        persist_history=lambda reason: events.append('persist'),
        start_hrs_planning=lambda: events.append('recompute'),
        return_to_lrs=lambda reason: events.append('return'))
    controller.advance_after_target = (
        lambda: HrsWorkflow(controller).finish_hrs_cycle())

    NavigationWorkflow(controller).finish_dwell(1)

    assert 'success' in events
    assert 'record_measurement' in events
    assert 'gmrf_update' in events
    assert 'publish_measurements' in events
    assert events.index('gmrf_update') < events.index('gmrf_finalize')
    assert events[-1] == 'recompute'


def test_hrs_cycle_returns_to_lrs_when_gmrf_recovery_fails():
    events = []
    controller = SimpleNamespace(
        repeat_after_hrs=True,
        active_hrs_target=SimpleNamespace(variable=3),
        hrs_route_targets=[SimpleNamespace(variable=3)],
        hrs_cycle_started_ns=0,
        hrs_cycles=0,
        hrs_cycles_in_alert=0,
        hrs_max_cycles_per_alert=10,
        hrs_response_threshold=0.5,
        hrs_confirmed_variable=None,
        hrs_confirmed_value=None,
        hrs_confirmed_timestamp=None,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        get_logger=lambda: SimpleNamespace(
            info=lambda message: None, warning=lambda message: None),
        finalize_hrs_gmrf_batch=lambda reason: False,
        publish_hrs_status=lambda: events.append('publish'),
        persist_history=lambda reason: events.append('persist'),
        return_to_lrs=lambda reason: events.append('return'))

    HrsWorkflow(controller).finish_hrs_cycle()

    assert events == ['publish', 'persist', 'return']


def test_hrs_planning_returns_to_lrs_when_candidates_are_exhausted():
    events = []
    controller = SimpleNamespace(
        repeat_after_hrs=True,
        planning_executor=SimpleNamespace(active_task=None),
        latest_pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=0.5, y=0.5))),
        hrs_candidate_variables=set(),
        hrs_ucb_k=0.2,
        hrs_distance_weight=0.03,
        hrs_cycles=0,
        hrs_route_targets=[],
        hrs_active_region_cells=set(),
        hrs_completed_region_cells=set(),
        hrs_region_ascent_floor=None,
        hrs_confirmed_variable=None,
        hrs_confirmed_value=None,
        hrs_confirmed_timestamp=None,
        hrs_manager=HrsManager(),
        sampling_distance=SimpleNamespace(
            distances_from=lambda first, targets: tuple(math.dist(first, point) for point in targets),
            distance=lambda first, second: 0.0),
        build_candidates=lambda: (),
        publish_phase=lambda phase: events.append('phase'),
        publish_candidates=lambda candidates, representatives, selected:
            events.append('candidates'),
        publish_empty_hrs_route=lambda: events.append('empty_route'),
        get_logger=lambda: SimpleNamespace(
            info=lambda message: events.append('info'),
            error=lambda message: events.append('error'),
            warning=lambda message: events.append('warning')),
        hrs_response_threshold=0.5,
        return_to_lrs=lambda reason: events.append(reason))

    HrsWorkflow(controller).start_hrs_planning()

    assert 'no eligible unmeasured HRS cells remain' in events[-1]


def test_hrs_planning_routes_only_one_highest_dd_ucb_cell():
    events = []
    candidate = HrsCandidate(
        variable=4, row=0, col=1, x=1.5, y=0.5,
        score=0.7, mean=0.6, variance=0.01)
    nearby = HrsCandidate(
        variable=5, row=0, col=0, x=0.6, y=0.5,
        score=0.6, mean=0.5, variance=0.01)
    far_peak = HrsCandidate(
        variable=6, row=0, col=5, x=5.5, y=0.5,
        score=0.9, mean=0.9, variance=0.01)
    candidates = (candidate, nearby, far_peak)
    controller = SimpleNamespace(
        repeat_after_hrs=True,
        planning_executor=SimpleNamespace(active_task=None),
        latest_pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=0.5, y=0.5))),
        hrs_candidate_variables={8},
        hrs_ucb_k=0.2,
        hrs_distance_weight=0.5,
        hrs_cycles=0,
        hrs_route_targets=[],
        hrs_active_region_cells=set(),
        hrs_completed_region_cells=set(),
        hrs_region_ascent_floor=None,
        hrs_manager=HrsManager(),
        sampling_distance=SimpleNamespace(
            distances_from=lambda first, targets: tuple(math.dist(first, point) for point in targets),
            distance=lambda first, second:
            ((first[0] - second[0]) ** 2 +
             (first[1] - second[1]) ** 2) ** 0.5),
        active_hrs_target=None,
        current_index=9,
        retry=2,
        hrs_cycle_started_ns=None,
        build_candidates=lambda: candidates,
        publish_phase=lambda phase: events.append(('phase', phase)),
        publish_candidates=lambda selected, representatives, target:
            events.append('candidates'),
        publish_hrs_route=lambda targets: events.append(
            ('route', tuple(target.variable for target in targets))),
        send_current_goal=lambda: events.append('send'),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=123)),
        get_logger=lambda: SimpleNamespace(
            info=lambda message: None, error=lambda message: None))

    HrsWorkflow(controller).start_hrs_planning()

    assert controller.active_hrs_target.variable == 6
    assert [target.variable for target in controller.hrs_route_targets] == [6]
    assert controller.hrs_candidate_variables == {4, 5, 6}
    assert controller.hrs_cycle_started_ns == 123
    assert events == [
        ('phase', 'HRS_PLANNING'), 'candidates', ('route', (6,)),
        ('phase', 'HRS_NAVIGATION'), 'send']


def test_each_completed_measurement_discards_route_and_replans():
    events = []
    controller = SimpleNamespace(
        repeat_after_hrs=True,
        active_hrs_target=SimpleNamespace(variable=3),
        hrs_route_targets=[SimpleNamespace(variable=3)],
        hrs_cycle_started_ns=0,
        hrs_cycles=0,
        hrs_cycles_in_alert=0,
        hrs_max_cycles_per_alert=10,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        get_logger=lambda: SimpleNamespace(
            info=lambda message: events.append(message),
            warning=lambda message: events.append(message)),
        finalize_hrs_gmrf_batch=lambda reason: events.append('gmrf') or True,
        publish_hrs_status=lambda: events.append('status'),
        persist_history=lambda reason: events.append('history'),
        start_hrs_planning=lambda: events.append('recompute'),
        return_to_lrs=lambda reason: events.append('return'))

    HrsWorkflow(controller).finish_hrs_cycle()

    assert controller.active_hrs_target is None
    assert controller.hrs_route_targets == []
    assert events[-1] == 'recompute'


@pytest.mark.parametrize('value', [True, False])
def test_repeat_after_hrs_configuration_roundtrip(value):
    config = ControllerConfig.from_mapping(controller_values(repeat_after_hrs=value))
    assert config.hrs.repeat_after_hrs is value
    assert config.flat_values()['repeat_after_hrs'] is value


@pytest.mark.parametrize('value', ['false', 'true', 0, 1, None])
def test_repeat_after_hrs_requires_boolean(value):
    with pytest.raises(ValueError, match='repeat_after_hrs'):
        ControllerConfig.from_mapping(controller_values(repeat_after_hrs=value))


@pytest.mark.parametrize('repeat', [True, False])
@pytest.mark.parametrize('reason', ['no candidates', 'GMRF update failed', 'maximum measurements reached'])
def test_hrs_end_repeat_policy_for_non_success(repeat, reason):
    calls = []
    controller = SimpleNamespace(
        repeat_after_hrs=repeat,
        get_logger=lambda: SimpleNamespace(warning=lambda msg: None),
        return_to_lrs=lambda msg: calls.append(('repeat', msg)),
        complete_mapping=lambda msg: calls.append(('complete', msg)))
    HrsWorkflow(controller).finish_hrs_search(reason)
    assert calls == [('repeat' if repeat else 'complete', 'HRS terminated: '+reason)]
