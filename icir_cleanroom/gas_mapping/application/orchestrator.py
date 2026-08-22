"""Top-level ownership of mapping workflow state transitions."""

import time

from ..models import MappingEventState, SourceTransitionState


class MappingOrchestrator:
    """Coordinate state owners without depending on ROS messages."""

    def __init__(self, phase_machine, navigation, lrs, hrs, peak,
                 event_state=None, source_state=None):
        self.phase_machine = phase_machine
        self.navigation = navigation
        self.lrs = lrs
        self.hrs = hrs
        self.peak = peak
        self.event = event_state or MappingEventState()
        self.source = source_state or SourceTransitionState()

    def begin_lrs_lap(self):
        lap = self.lrs.begin_lap()
        self.navigation.reset_progress()
        self.hrs.reset_search()
        self.peak.reset()
        self.event.snapshot_pending = True
        self.event.reset_measurements()
        if not self.event.event_id:
            self.event.event_id = f'controller-source-{time.time_ns()}'
        return lap

    def begin_source_transition(self, reason):
        self.source.generation += 1
        self.source.reason = str(reason)
        self.source.pending = True
        self.source.timed_out = False
        return self.source.generation

    def record_event_measurement(self, variable, value, variable_cells):
        variable = int(variable)
        value = float(value)
        self.event.sampled_variables.add(variable)
        self.event.measured_by_variable[variable] = value
        best = min(
            self.event.measured_by_variable,
            key=lambda item: (
                -float(self.event.measured_by_variable[item]),
                int(variable_cells[item][0]), int(variable_cells[item][1])))
        self.event.best_variable = int(best)
        self.event.best_value = float(
            self.event.measured_by_variable[best])
        return self.event.best_variable, self.event.best_value

    def finish_source_transition(self, source_changed, timed_out=False):
        self.source.pending = False
        self.source.timed_out = bool(timed_out)
        if source_changed:
            self.event.event_id = ''


__all__ = ['MappingOrchestrator']
