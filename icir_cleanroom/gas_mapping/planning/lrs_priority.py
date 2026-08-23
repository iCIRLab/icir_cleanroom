"""Reward-prioritized prefix planner with complete LRS coverage."""

from dataclasses import dataclass
from itertools import combinations
import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .exact_path import solve_held_karp_free_end_path


@dataclass(frozen=True)
class LrsRewardPoint:
    """One active LRS waypoint and its normalized reward components."""

    variable: int
    row: int
    col: int
    x: float
    y: float
    reward: float
    recurrence: float = 0.0
    severity: float = 0.0
    staleness: float = 0.0
    uncertainty: float = 0.0
    mandatory: bool = False


@dataclass
class LrsPriorityRoute:
    """Complete route with an exact high-reward prefix."""

    points: list
    priority_points: list
    reward: float
    length: float
    baseline_length: float
    length_ratio: float
    priority_count: int
    total_combinations: int
    evaluated_routes: int
    feasible_routes: int
    fallback: bool
    status: str
    planning_seconds: float


def _distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _point_xy(point):
    return float(point.x), float(point.y)


def _point_key(point):
    return int(point.row), int(point.col), int(point.variable)


def _route_length(points, start_xy, close_cycle=True, distance_fn=None):
    distance_fn = distance_fn or _distance
    if not points:
        return 0.0
    total = distance_fn(tuple(start_xy), _point_xy(points[0]))
    for first, second in zip(points, points[1:]):
        total += distance_fn(_point_xy(first), _point_xy(second))
    if close_cycle:
        total += distance_fn(
            _point_xy(points[-1]), _point_xy(points[0]))
    return float(total)


def _best_cycle_order(points, start_xy, distance_fn=None):
    """Rotate and orient a cycle to minimize travel from the current pose."""
    if not points:
        return []
    source = list(points)
    best = None
    for direction in (source, list(reversed(source))):
        for offset in range(len(direction)):
            ordered = direction[offset:] + direction[:offset]
            key = (
                _route_length(
                    ordered, start_xy, distance_fn=distance_fn),
                tuple(_point_key(point) for point in ordered),
            )
            if best is None or key < best[0]:
                best = (key, ordered)
    return best[1]


def _best_coverage_suffix(
        prefix, active_cycle, start_xy, distance_fn=None):
    """Append every non-prefix point in the best cycle order exactly once."""
    selected = {_point_key(point) for point in prefix}
    remaining = [
        point for point in active_cycle if _point_key(point) not in selected]
    if not remaining:
        return list(prefix), _route_length(
            prefix, start_xy, distance_fn=distance_fn)

    best = None
    for direction in (remaining, list(reversed(remaining))):
        for offset in range(len(direction)):
            suffix = direction[offset:] + direction[:offset]
            ordered = list(prefix) + suffix
            key = (
                _route_length(
                    ordered, start_xy, distance_fn=distance_fn),
                tuple(_point_key(point) for point in ordered),
            )
            if best is None or key < best[0]:
                best = (key, ordered)
    return best[1], float(best[0][0])


def _candidate_pool(points, candidate_count):
    ranked = sorted(
        points,
        key=lambda point: (
            not bool(point.mandatory), -float(point.reward), *_point_key(point)))
    mandatory = [point for point in ranked if point.mandatory]
    optional = [point for point in ranked if not point.mandatory]
    size = max(int(candidate_count), len(mandatory))
    return mandatory + optional[:max(0, size - len(mandatory))]


def _ranked_combinations(candidates, visit_count):
    mandatory_keys = {
        _point_key(point) for point in candidates if point.mandatory}
    ranked = []
    for selected in combinations(candidates, visit_count):
        selected_keys = {_point_key(point) for point in selected}
        if not mandatory_keys.issubset(selected_keys):
            continue
        reward = float(sum(float(point.reward) for point in selected))
        key = tuple(sorted(_point_key(point) for point in selected))
        ranked.append((-reward, key, selected, reward))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked


def solve_lrs_priority_route(active_cycle, start_xy, candidate_count=15,
                             priority_count=6, length_ratio_limit=1.10,
                             distance_fn=None):
    """Plan a reward-maximal prefix while retaining full cyclic coverage.

    Candidate combinations are examined by decreasing reward. Each prefix is
    solved exactly with Held--Karp, then joined to the best rotation and
    direction of the remaining coverage cycle. Equal-reward feasible sets are
    resolved by complete-route length and stable cell order.
    """
    started = time.monotonic()
    points = list(active_cycle)
    if len(points) < 2:
        raise ValueError('adaptive LRS requires at least two active points')
    if int(candidate_count) <= 0 or int(priority_count) <= 0:
        raise ValueError('candidate and priority counts must be positive')
    if not math.isfinite(float(length_ratio_limit)) or not (
            float(length_ratio_limit) >= 1.0):
        raise ValueError('length_ratio_limit must be finite and at least 1')
    keys = [_point_key(point) for point in points]
    if len(set(keys)) != len(keys):
        raise ValueError('active LRS points must have unique stable keys')

    baseline = _best_cycle_order(points, start_xy, distance_fn)
    baseline_length = _route_length(
        baseline, start_xy, distance_fn=distance_fn)
    length_limit = baseline_length * float(length_ratio_limit)
    candidates = _candidate_pool(points, candidate_count)
    mandatory_count = sum(1 for point in candidates if point.mandatory)
    maximum = min(int(priority_count), len(candidates))
    total_combinations = 0
    evaluated = 0
    feasible = 0

    for count in range(maximum, max(0, mandatory_count - 1), -1):
        ranked = _ranked_combinations(candidates, count)
        total_combinations += len(ranked)
        best = None
        best_reward = None
        for negative_reward, cell_key, selected_tuple, reward in ranked:
            if best_reward is not None and reward < best_reward - 1.0e-12:
                break
            evaluated += 1
            solved = solve_held_karp_free_end_path(
                list(selected_tuple), start_xy, distance_fn)
            if solved is None:
                continue
            prefix, _ = solved
            ordered, length = _best_coverage_suffix(
                prefix, points, start_xy, distance_fn)
            if length > length_limit + 1.0e-9:
                continue
            feasible += 1
            selection_key = (
                length,
                tuple(_point_key(point) for point in prefix),
                cell_key,
            )
            if best is None or selection_key < best[0]:
                best = (selection_key, ordered, prefix, length, reward)
                best_reward = reward
        if best is not None:
            ratio = (best[3] / baseline_length
                     if baseline_length > 0.0 else 1.0)
            return LrsPriorityRoute(
                points=best[1], priority_points=best[2], reward=best[4],
                length=best[3], baseline_length=baseline_length,
                length_ratio=ratio, priority_count=len(best[2]),
                total_combinations=total_combinations,
                evaluated_routes=evaluated, feasible_routes=feasible,
                fallback=False, status='PRIORITY_EXACT',
                planning_seconds=time.monotonic() - started)

    return lrs_coverage_fallback(
        points, start_xy, baseline=baseline,
        baseline_length=baseline_length,
        total_combinations=total_combinations,
        evaluated_routes=evaluated, feasible_routes=feasible,
        planning_started=started, distance_fn=distance_fn)


def lrs_coverage_fallback(active_cycle, start_xy, baseline=None,
                          baseline_length=None, total_combinations=0,
                          evaluated_routes=0, feasible_routes=0,
                          planning_started=None, distance_fn=None):
    """Return the shortest rotation/orientation of the coverage cycle."""
    points = list(active_cycle)
    if not points:
        raise ValueError('LRS coverage fallback requires active points')
    if planning_started is None:
        planning_started = time.monotonic()
    if baseline is None:
        baseline = _best_cycle_order(points, start_xy, distance_fn)
    if baseline_length is None:
        baseline_length = _route_length(
            baseline, start_xy, distance_fn=distance_fn)
    return LrsPriorityRoute(
        points=baseline, priority_points=[], reward=0.0,
        length=baseline_length, baseline_length=baseline_length,
        length_ratio=1.0, priority_count=0,
        total_combinations=total_combinations,
        evaluated_routes=evaluated_routes, feasible_routes=feasible_routes,
        fallback=True, status='BASE_COVERAGE_FALLBACK',
        planning_seconds=time.monotonic() - planning_started)




def distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])

def history_adjusted_cycle(base_points, history_points, replace_radius,
                           start_xy, distance_fn=None):
    """Replace nearby base points, insert distant points, and rotate a cycle.

    Returns ``(ordered_points, replacements, additions)``. Points are ``(x,y)``
    tuples. A large assignment penalty first maximizes valid one-to-one
    replacements and then minimizes their total distance.
    """
    route = [tuple(point) for point in base_points]
    history = [tuple(point) for point in history_points]
    distance_fn = distance_fn or distance
    replacements = []
    additions = []
    if not route:
        return history, replacements, history

    replaced_history = set()
    if history:
        distances = np.asarray([
            [distance_fn(hotspot, base) for base in route]
            for hotspot in history
        ], dtype=float)
        penalty = 1.0e6
        assignment_cost = distances + (distances > replace_radius) * penalty
        history_indices, base_indices = linear_sum_assignment(assignment_cost)
        for history_index, base_index in zip(history_indices, base_indices):
            separation = float(distances[history_index, base_index])
            if separation <= replace_radius:
                original = route[base_index]
                route[base_index] = history[history_index]
                replaced_history.add(int(history_index))
                replacements.append(
                    (original, history[history_index], separation))

    for history_index, hotspot in enumerate(history):
        if history_index in replaced_history:
            continue
        best = None
        for index, first in enumerate(route):
            second = route[(index + 1) % len(route)]
            increase = (
                distance_fn(first, hotspot) +
                distance_fn(hotspot, second) -
                distance_fn(first, second))
            key = (increase, index)
            if best is None or key < best[0]:
                best = (key, index + 1)
        route.insert(best[1], hotspot)
        additions.append(hotspot)

    start_index = min(
        range(len(route)),
        key=lambda index: (distance_fn(route[index], start_xy), index))
    route = route[start_index:] + route[:start_index]
    return route, replacements, additions


__all__ = [
    'LrsPriorityRoute', 'LrsRewardPoint', 'history_adjusted_cycle',
    'lrs_coverage_fallback', 'solve_lrs_priority_route',
]
