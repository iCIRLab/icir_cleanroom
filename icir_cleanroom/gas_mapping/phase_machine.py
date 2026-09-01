"""Explicit phase transitions for the mapping workflow."""

from .models import MappingPhase


ALLOWED_TRANSITIONS = {
    MappingPhase.WAITING: {
        MappingPhase.WAITING, MappingPhase.LRS_PLANNING,
        MappingPhase.COMPLETE},
    MappingPhase.LRS_PLANNING: {
        MappingPhase.LRS_PLANNING, MappingPhase.LRS,
        MappingPhase.COMPLETE},
    MappingPhase.LRS: {
        MappingPhase.LRS, MappingPhase.LRS_PLANNING,
        MappingPhase.HRS_PLANNING, MappingPhase.COMPLETE},
    MappingPhase.HRS_PLANNING: {
        MappingPhase.HRS_PLANNING, MappingPhase.HRS_NAVIGATION,
        MappingPhase.LRS_PLANNING,
        MappingPhase.COMPLETE},
    MappingPhase.HRS_NAVIGATION: {
        MappingPhase.HRS_NAVIGATION, MappingPhase.HRS_PLANNING,
        MappingPhase.SOURCE_TRANSITION, MappingPhase.LRS_PLANNING,
        MappingPhase.COMPLETE},
    MappingPhase.SOURCE_TRANSITION: {
        MappingPhase.SOURCE_TRANSITION, MappingPhase.LRS_PLANNING,
        MappingPhase.COMPLETE},
    MappingPhase.COMPLETE: {MappingPhase.COMPLETE},
}


class InvalidPhaseTransition(ValueError):
    pass


class PhaseMachine:
    def __init__(self, initial=MappingPhase.WAITING):
        self._phase = MappingPhase(initial)
        self._generation = 0

    @property
    def phase(self):
        return self._phase

    @property
    def generation(self):
        return self._generation

    def transition(self, target):
        target = MappingPhase(target)
        if target not in ALLOWED_TRANSITIONS[self._phase]:
            raise InvalidPhaseTransition(
                f'{self._phase.value} -> {target.value} is not allowed')
        if target != self._phase:
            self._generation += 1
        self._phase = target
        return self._phase

    def accepts(self, phase, generation):
        return self._phase == MappingPhase(phase) and (
            self._generation == int(generation))


__all__ = [
    'ALLOWED_TRANSITIONS', 'InvalidPhaseTransition', 'PhaseMachine',
]
