"""Dwell-sample session management independent from ROS timers."""

from dataclasses import dataclass, field
import math

from ..models import GasMeasurement, MappingPhase, Pose2D


@dataclass
class DwellSession:
    generation: int
    pose: object
    phase: object
    target_variable: int | None
    values: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class DwellResult:
    generation: int
    pose: object
    phase: object
    target_variable: int | None
    mean: float | None
    sample_count: int


class MeasurementManager:
    def __init__(self):
        self._generation = 0
        self.active = None
        self.measurements = []

    @property
    def measuring(self):
        return self.active is not None

    def start(self, pose, phase, target_variable=None):
        self._generation += 1
        self.active = DwellSession(
            self._generation, pose, phase, target_variable)
        return self._generation

    def add_sample(self, value):
        if self.active is not None:
            self.active.values.append(float(value))
            return True
        return False

    def finish(self, generation):
        if self.active is None or self.active.generation != int(generation):
            return None
        session = self.active
        self.active = None
        mean = None
        if session.values:
            mean = float(sum(session.values) / len(session.values))
        return DwellResult(
            session.generation, session.pose, session.phase,
            session.target_variable, mean, len(session.values))

    def cancel(self):
        self._generation += 1
        self.active = None

    def commit(self, result, variable, timestamp):
        """Convert a completed ROS-shaped pose into a domain measurement."""
        pose = result.pose.pose
        orientation = pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z +
                   orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y +
                         orientation.z * orientation.z))
        measurement = GasMeasurement(
            pose=Pose2D(float(pose.position.x), float(pose.position.y), yaw),
            value=float(result.mean), variable=int(variable),
            target_variable=result.target_variable,
            phase=MappingPhase(result.phase), timestamp=float(timestamp),
            sample_count=int(result.sample_count))
        self.measurements.append(measurement)
        return measurement


__all__ = ['DwellResult', 'DwellSession', 'MeasurementManager']
