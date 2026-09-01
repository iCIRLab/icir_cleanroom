"""ROS-independent value objects and workflow runtime state."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Set


class MappingPhase(str, Enum):
    WAITING = 'WAITING'
    LRS_PLANNING = 'LRS_PLANNING'
    LRS = 'LRS'
    HRS_PLANNING = 'HRS_PLANNING'
    HRS_NAVIGATION = 'HRS_NAVIGATION'
    SOURCE_TRANSITION = 'SOURCE_TRANSITION'
    COMPLETE = 'COMPLETE'

    def __str__(self):
        return self.value


class PlanningKind(str, Enum):
    LRS = 'LRS'
    HRS = 'HRS'

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class GasMeasurement:
    pose: Pose2D
    value: float
    variable: int
    target_variable: Optional[int]
    phase: MappingPhase
    timestamp: float
    sample_count: int


@dataclass
class MappingEventState:
    event_id: str = ''
    sampled_variables: Set[int] = field(default_factory=set)
    measured_by_variable: Dict[int, float] = field(default_factory=dict)
    snapshot_pending: bool = False

    def reset_measurements(self):
        self.sampled_variables.clear()
        self.measured_by_variable.clear()


@dataclass
class NavigationState:
    generation: int = 0
    index: int = 0
    retry_count: int = 0
    returning: bool = False
    active_goal_id: Optional[int] = None

    def reset_progress(self):
        self.generation += 1
        self.index = 0
        self.retry_count = 0
        self.returning = False
        self.active_goal_id = None


@dataclass
class LrsRuntimeState:
    lap: int = 0
    base_goals: list = field(default_factory=list)
    goals: list = field(default_factory=list)
    status_values: list = field(default_factory=list)
    active_history_regions: list = field(default_factory=list)
    reward_values: object = None
    reward_points: list = field(default_factory=list)
    priority_candidates: list = field(default_factory=list)
    priority_points: list = field(default_factory=list)
    lap_max_concentration: float = 0.0
    hazard_detected: bool = False


@dataclass
class HrsRuntimeState:
    active_route: object = None
    cycles: int = 0
    cycles_in_alert: int = 0
    batch_successes: int = 0
    cycle_started_ns: Optional[int] = None
    failure_counts: Dict[int, int] = field(default_factory=dict)
    unreachable_variables: Set[int] = field(default_factory=set)
    dirty: bool = False

    def reset_search(self):
        self.cycles = 0
        self.cycles_in_alert = 0
        self.batch_successes = 0
        self.failure_counts.clear()
        self.unreachable_variables.clear()
        self.dirty = False


@dataclass
class SourceTransitionState:
    generation: int = 0
    reason: str = ''
    pending: bool = False
    timed_out: bool = False


@dataclass(frozen=True)
class PlanningTask:
    generation: int
    kind: PlanningKind
    phase: MappingPhase
    lap: int
    event_id: str
    context: object


class StateField:
    """Descriptor used while legacy method names migrate to state owners."""

    def __init__(self, owner_name, field_name):
        self.owner_name = owner_name
        self.field_name = field_name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(getattr(instance, self.owner_name), self.field_name)

    def __set__(self, instance, value):
        setattr(getattr(instance, self.owner_name), self.field_name, value)


__all__ = [
    'GasMeasurement', 'HrsRuntimeState', 'LrsRuntimeState',
    'MappingEventState', 'MappingPhase', 'NavigationState',
    'PlanningKind', 'PlanningTask', 'Pose2D',
    'SourceTransitionState', 'StateField',
]
