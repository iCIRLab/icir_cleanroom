"""Tests for typed configuration and explicit runtime state."""

import pytest

from icir_cleanroom.gas_mapping.config import (
    ControllerConfig, DEFAULT_CONTROLLER_PARAMETERS)
from icir_cleanroom.gas_mapping.application.hrs import HrsManager
from icir_cleanroom.gas_mapping.application.lrs import LrsManager
from icir_cleanroom.gas_mapping.application.navigation import NavigationManager
from icir_cleanroom.gas_mapping.application.orchestrator import (
    MappingOrchestrator)
from icir_cleanroom.gas_mapping.application.peak_confirmation import (
    PeakConfirmationManager)
from icir_cleanroom.gas_mapping.models import (
    HrsRuntimeState, MappingPhase, NavigationState)
from icir_cleanroom.gas_mapping.phase_machine import (
    InvalidPhaseTransition, PhaseMachine)


def test_typed_config_round_trips_all_ros_parameter_defaults():
    config = ControllerConfig.from_mapping(DEFAULT_CONTROLLER_PARAMETERS)
    assert config.flat_values() == DEFAULT_CONTROLLER_PARAMETERS
    assert config.hrs.hrs_visit_count == 10
    assert config.gmrf.gabp_damping == 0.5


def test_config_rejects_invalid_cross_field_values():
    values = dict(DEFAULT_CONTROLLER_PARAMETERS)
    values['lrs_priority_count'] = values['lrs_priority_candidate_count'] + 1
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
        cycles=3, failure_counts={7: 2}, unreachable_variables={7}, dirty=True)
    hrs.reset_search()
    assert hrs.cycles == 0
    assert hrs.failure_counts == {}
    assert hrs.unreachable_variables == set()
    assert hrs.dirty is False


def test_orchestrator_owns_deterministic_event_best_measurement():
    phases = PhaseMachine()
    event_orchestrator = MappingOrchestrator(
        phases, NavigationManager(), LrsManager(), HrsManager(),
        PeakConfirmationManager())
    cells = [(1, 1), (0, 2), (0, 1)]
    event_orchestrator.record_event_measurement(0, 0.7, cells)
    best, value = event_orchestrator.record_event_measurement(2, 0.7, cells)

    assert (best, value) == (2, 0.7)
    assert event_orchestrator.event.sampled_variables == {0, 2}
