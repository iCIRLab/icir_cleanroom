"""Tests for typed configuration and explicit runtime state."""

from types import SimpleNamespace

import pytest

from icir_cleanroom.gas_mapping.application.hrs import HrsManager
from icir_cleanroom.gas_mapping.application.lrs import LrsManager
from icir_cleanroom.gas_mapping.application.navigation import NavigationManager
from icir_cleanroom.gas_mapping.application.orchestrator import (
    MappingOrchestrator)
from icir_cleanroom.gas_mapping.config import (
    ControllerConfig, DEFAULT_CONTROLLER_PARAMETERS)
from icir_cleanroom.gas_mapping.models import (
    HrsRuntimeState, MappingPhase, NavigationState)
from icir_cleanroom.gas_mapping.phase_machine import (
    InvalidPhaseTransition, PhaseMachine)
from icir_cleanroom.gas_mapping.ros.hrs_workflow import HrsWorkflow
from icir_cleanroom.gas_mapping.ros.navigation_workflow import (
    NavigationWorkflow)


def logger(events):
    return SimpleNamespace(
        info=lambda message: events.append('info'),
        warning=lambda message: events.append('warning'),
        error=lambda message: events.append('error'))


def test_typed_config_round_trips_all_ros_parameter_defaults():
    config = ControllerConfig.from_mapping(DEFAULT_CONTROLLER_PARAMETERS)
    assert config.flat_values() == DEFAULT_CONTROLLER_PARAMETERS
    assert config.hrs.hrs_centroid_threshold == 0.0
    assert config.hrs.hrs_response_threshold == 0.5
    assert config.gmrf.gabp_damping == 0.5


def test_config_rejects_invalid_cross_field_values():
    values = dict(DEFAULT_CONTROLLER_PARAMETERS)
    values['lrs_priority_count'] = values['lrs_priority_candidate_count'] + 1
    with pytest.raises(ValueError):
        ControllerConfig.from_mapping(values)

    for name, value in (
            ('hrs_response_threshold', 1.01),
            ('hrs_centroid_threshold', -0.01)):
        values = dict(DEFAULT_CONTROLLER_PARAMETERS)
        values[name] = value
        with pytest.raises(ValueError):
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


def test_runtime_reset_methods_advance_or_clear_owned_state():
    navigation = NavigationState(index=4, retry_count=2, returning=True)
    navigation.reset_progress()
    assert (navigation.generation, navigation.index,
            navigation.retry_count, navigation.returning) == (1, 0, 0, False)

    hrs = HrsRuntimeState(
        active_target=object(), centroid=(1.0, 2.0), cycles=3,
        search_started_ns=123, failure_counts={7: 2},
        unreachable_variables={7}, dirty=True)
    hrs.reset_search()
    assert hrs.active_target is None
    assert hrs.centroid is None
    assert hrs.cycles == 0
    assert hrs.search_started_ns is None
    assert hrs.failure_counts == {}
    assert hrs.unreachable_variables == set()
    assert hrs.dirty is False


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


def test_complete_hrs_measurement_time_is_logged_once():
    messages = []
    now = SimpleNamespace(nanoseconds=2_000_000_000)
    controller = SimpleNamespace(
        hrs_search_started_ns=None,
        hrs_cycles_in_alert=3,
        lrs_lap=2,
        get_clock=lambda: SimpleNamespace(now=lambda: now),
        get_logger=lambda: SimpleNamespace(
            info=lambda message: messages.append(message)))
    workflow = HrsWorkflow(controller)

    workflow.begin_hrs_measurement_timing()
    now.nanoseconds = 7_250_000_000
    elapsed = workflow.finish_hrs_measurement_timing(
        'response threshold reached')

    assert elapsed == pytest.approx(5.25)
    assert controller.hrs_search_started_ns is None
    assert messages == [
        '=== HRS 전체 측정 시작: LRS lap=2 ===',
        '=== HRS 전체 측정 종료: outcome=response threshold reached, '
        'elapsed=5.250s ===',
    ]
    assert workflow.finish_hrs_measurement_timing('duplicate') is None
    assert len(messages) == 2


@pytest.mark.parametrize(
    ('completed_cycles', 'expected_action'),
    [(0, 'replan'), (9, 'return')])
def test_single_cell_hrs_cycle_replans_until_maximum(
        completed_cycles, expected_action):
    events = []
    controller = SimpleNamespace(
        active_hrs_target=SimpleNamespace(variable=3),
        hrs_cycle_started_ns=0,
        hrs_cycles=completed_cycles,
        hrs_cycles_in_alert=completed_cycles,
        hrs_max_cycles_per_alert=10,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        get_logger=lambda: logger(events),
        finalize_hrs_gmrf_batch=lambda reason: events.append('finalize') or True,
        publish_hrs_status=lambda: events.append('publish'),
        persist_history=lambda reason: events.append('persist'),
        start_hrs_planning=lambda: events.append('replan'),
        return_to_lrs=lambda reason: events.append('return'))

    workflow = HrsWorkflow(controller)
    workflow.start_hrs_planning = lambda: events.append('replan')
    workflow.finish_hrs_cycle()

    assert controller.active_hrs_target is None
    expected = ['info', 'finalize', 'publish', 'persist', expected_action]
    if expected_action == 'return':
        expected.insert(-1, 'warning')
    assert events == expected


def test_confirmed_response_is_persisted_before_source_transition():
    events = []
    history = SimpleNamespace(record_confirmed_event=lambda *args, **kwargs:
                              events.append(('record', args, kwargs)) or True)
    controller = SimpleNamespace(
        hrs_response_threshold=0.5,
        hrs_manager=HrsManager(),
        gmrf=SimpleNamespace(var_cells=[(2, 3)]),
        history=history,
        current_event_id='event-1',
        hrs_gmrf_dirty=False,
        get_logger=lambda: logger(events),
        publish_history=lambda: events.append('publish'),
        persist_history=lambda reason: events.append('persist') or True,
        start_source_transition=lambda reason: events.append('transition'))

    workflow = HrsWorkflow(controller)
    assert not workflow.confirm_hrs_response(0, 0.4999, 10.0)
    events.clear()
    assert workflow.confirm_hrs_response(0, 0.5, 11.0)

    assert events[0][0] == 'record'
    assert events[0][1][0:6] == ('event-1', 2, 3, 0.5, 0.5, 11.0)
    assert events[-4:] == ['publish', 'persist', 'warning', 'transition']


@pytest.mark.parametrize(
    ('value', 'stops_iteration'), [(0.4999, False), (0.5, True)])
def test_hrs_dwell_stops_iteration_at_response_threshold(
        value, stops_iteration):
    events = []
    pose = SimpleNamespace(position=SimpleNamespace(x=0.5, y=0.5))
    result = SimpleNamespace(
        phase=MappingPhase.HRS_NAVIGATION, mean=value,
        pose=SimpleNamespace(pose=pose), sample_count=3)
    measurement = SimpleNamespace(timestamp=11.0)
    measurement_manager = SimpleNamespace(
        finish=lambda generation: result,
        commit=lambda completed, variable, timestamp: measurement)
    gmrf = SimpleNamespace(
        set_observation=lambda x, y, measured: (0, 0),
        variable_at=lambda row, col: 0)
    controller = SimpleNamespace(
        phase=MappingPhase.HRS_NAVIGATION,
        measurement_manager=measurement_manager,
        dwell_timer=None,
        gmrf=gmrf,
        history=SimpleNamespace(update=lambda *args: events.append('history')),
        active_hrs_target=SimpleNamespace(variable=0, row=0, col=0),
        current_index=0,
        hrs_manager=SimpleNamespace(
            record_success=lambda variable: events.append('success')),
        hrs_gmrf_dirty=False,
        hrs_response_threshold=0.5,
        get_logger=lambda: logger(events),
        publish_history=lambda: events.append('publish_history'),
        record_event_measurement=lambda variable, measured:
            events.append('record_measurement'),
        update_gmrf=lambda reason, compare_with_cg=False: True,
        publish_measurements=lambda: events.append('publish_measurements'),
        confirm_hrs_response=lambda variable, measured, timestamp:
            measured >= 0.5,
        advance_after_target=lambda: events.append('advance'))

    NavigationWorkflow(controller).finish_dwell(1)

    assert ('advance' in events) is not stops_iteration


def test_hrs_cycle_returns_to_lrs_when_gmrf_recovery_fails():
    events = []
    controller = SimpleNamespace(
        active_hrs_target=SimpleNamespace(variable=3),
        hrs_cycle_started_ns=0,
        hrs_cycles=0,
        hrs_cycles_in_alert=0,
        hrs_max_cycles_per_alert=10,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        get_logger=lambda: logger(events),
        finalize_hrs_gmrf_batch=lambda reason: False,
        publish_hrs_status=lambda: events.append('publish'),
        persist_history=lambda reason: events.append('persist'),
        return_to_lrs=lambda reason: events.append('return'))

    HrsWorkflow(controller).finish_hrs_cycle()

    assert events == [
        'info', 'publish', 'persist', 'warning', 'return']


def test_hrs_planning_returns_to_lrs_without_centroid_mass():
    events = []
    selection = SimpleNamespace(centroid=None, target=None)
    controller = SimpleNamespace(
        planning_executor=SimpleNamespace(active_task=None),
        hrs_manager=SimpleNamespace(select_target=lambda *args: selection),
        gmrf=object(), sampled_variables=set(),
        hrs_centroid_threshold=0.15,
        navigation_goal_variables={0}, hrs_centroid=None,
        active_hrs_target=None, hrs_search_started_ns=None,
        lrs_lap=1,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        publish_phase=lambda phase: events.append('phase'),
        publish_centroid=lambda centroid, target: events.append('centroid'),
        publish_empty_hrs_route=lambda: events.append('empty_route'),
        get_logger=lambda: logger(events),
        return_to_lrs=lambda reason: events.append('return'))

    HrsWorkflow(controller).start_hrs_planning()

    assert events == [
        'phase', 'info', 'centroid', 'empty_route', 'warning', 'info',
        'return']
