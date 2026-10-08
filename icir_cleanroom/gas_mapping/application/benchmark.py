"""Fixed benchmark scales and evaluation-only, frozen navigation references."""
from dataclasses import dataclass
import json
import math

import numpy as np

from ..mapping.domains import mask_contains_world


TIME_REFERENCE_SECONDS = 160.0
ERROR_REFERENCE_METERS = 0.5
REFERENCE_TOLERANCE_METERS = 1e-9

EVALUATION_FIELDS = (
    'evaluation_reference_candidates', 'evaluation_reference_adjusted',
    'evaluation_reference_reason', 'evaluation_reference_distance_m',
    'evaluation_available', 'evaluation_reason', 'adjusted_source_position_error')
SOURCE_TRUTH_FIELDS = (
    'source_sigma', 'source_strength', 'source_mode',
    'source_detection_threshold', 'source_detection_point_count',
    'source_generation_attempts', 'source_rejected_detection',
    'source_rejected_separation', 'source_rejected_out_of_bounds',
    'source_generation_exhausted')
TIME_FIELDS = (
    'lrs_seconds', 'cumulative_lrs_seconds', 'experiment_seconds',
    'wall_lrs_seconds', 'wall_cumulative_lrs_seconds', 'wall_experiment_seconds',
    'experiment_start_ros_seconds', 'experiment_ros_clock_valid',
    'experiment_completed')


def valid_xy(xy):
    return xy is not None and len(xy) == 2 and all(math.isfinite(float(v)) for v in xy)


@dataclass(frozen=True)
class EvaluationReference:
    source: object = None
    candidates: tuple = ()
    adjusted: object = None
    reason: str = 'missing_map'
    distance: object = None

    def evaluate(self, estimate, source):
        reason = self.reason
        error = None
        if not valid_xy(source):
            reason = 'missing_source'
        elif not valid_xy(self.source):
            reason = 'missing_source_at_start'
        elif math.dist(self.source, source) > REFERENCE_TOLERANCE_METERS:
            reason = 'source_changed'
        elif not self.candidates:
            pass
        elif not valid_xy(estimate):
            reason = 'missing_estimate'
        else:
            error = min(math.dist(estimate, p) for p in self.candidates)
            reason = 'available'
        return dict(
            evaluation_reference_candidates=json.dumps(self.candidates, allow_nan=False),
            evaluation_reference_adjusted=self.adjusted,
            evaluation_reference_reason=self.reason,
            evaluation_reference_distance_m=self.distance,
            evaluation_available=error is not None, evaluation_reason=reason,
            adjusted_source_position_error=error)


@dataclass(frozen=True)
class EvaluationDomain:
    geometry: object
    mask: object
    candidates: tuple

    @classmethod
    def snapshot(cls, geometry, mask, field_geometry):
        if geometry is None or mask is None or field_geometry is None:
            return None
        frozen = np.asarray(mask, dtype=bool).copy()
        if frozen.shape != (geometry.height, geometry.width):
            raise ValueError('evaluation navigation mask shape mismatch')
        frozen.flags.writeable = False
        candidates = tuple(
            xy for r in range(field_geometry.height) for c in range(field_geometry.width)
            for xy in (field_geometry.cell_center(r, c),)
            if mask_contains_world(frozen, geometry, *xy))
        return cls(geometry, frozen, candidates)

    def reference(self, source):
        if not valid_xy(source):
            return EvaluationReference(reason='missing_source')
        source = tuple(map(float, source))
        if not self.candidates:
            return EvaluationReference(source=source, reason='no_reachable_cells')
        if mask_contains_world(self.mask, self.geometry, *source):
            return EvaluationReference(source, (source,), False, 'source_reachable', 0.0)
        distances = [math.dist(source, xy) for xy in self.candidates]
        nearest = min(distances)
        candidates = tuple(xy for xy, distance in zip(self.candidates, distances)
                           if abs(distance-nearest) <= REFERENCE_TOLERANCE_METERS)
        return EvaluationReference(source, candidates, True, 'source_outside_goal_domain', nearest)


def cost(seconds, error, time_weight=0.6):
    if (not all(math.isfinite(v) and v >= 0 for v in (seconds, error))
            or not 0 <= time_weight <= 1):
        raise ValueError('benchmark inputs must be finite, nonnegative; weight in [0, 1]')
    return time_weight * seconds / TIME_REFERENCE_SECONDS + (1-time_weight) * error / ERROR_REFERENCE_METERS


def score(costs):
    values = list(costs)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('score needs a complete, nonnegative finite cost list')
    return 100.0 / (1.0 + math.fsum(values) / len(values))


def validate_evaluation(row):
    """Check a serialized v5 result without changing legacy raw error semantics."""
    try:
        candidates = json.loads(row['evaluation_reference_candidates'])
        if any(not valid_xy(xy) for xy in candidates):
            return False
        available = str(row['evaluation_available']).lower() == 'true'
        error = row['adjusted_source_position_error']
        if not available:
            return error in ('', None) and row['evaluation_reason'] != 'available'
        estimate = (float(row['estimated_source_x']), float(row['estimated_source_y']))
        return (bool(candidates) and row['evaluation_reason'] == 'available'
                and math.isclose(float(error), min(math.dist(estimate, p) for p in candidates),
                                 rel_tol=1e-9, abs_tol=1e-7))
    except (KeyError, TypeError, ValueError):
        return False


def validate_timing(row):
    """Validate extra serialized timing fields; missing historical values stay blank."""
    try:
        numeric = [k for k in TIME_FIELDS if k.endswith('seconds') and k != 'experiment_start_ros_seconds']
        values = {k: float(row[k]) for k in numeric if row.get(k) not in (None, '')}
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            return False
        for prefix in ('', 'wall_'):
            lap, cumulative, total = (values.get(prefix+k) for k in ('lrs_seconds','cumulative_lrs_seconds','experiment_seconds'))
            if lap is not None and cumulative is not None and lap > cumulative+1e-7:
                return False
            hrs = row.get('wall_total_seconds' if prefix else 'total_seconds')
            if total is not None and cumulative is not None and hrs not in (None, ''):
                if total+1e-7 < cumulative+float(hrs):
                    return False
        if str(row.get('experiment_ros_clock_valid')).lower() == 'false':
            return row.get('experiment_seconds') in (None, '') and row.get('cumulative_lrs_seconds') in (None, '')
        return True
    except (ValueError, TypeError):
        return False
