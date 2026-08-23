"""Lock down deterministic exact-path and LRS/HRS planning behavior."""

import math

from icir_cleanroom.gas_mapping.planning.hrs import (
    HrsCell, solve_held_karp_free_end_path, solve_held_karp_open_path,
    solve_hrs_p2)
from icir_cleanroom.gas_mapping.planning.lrs_priority import (
    LrsRewardPoint, history_adjusted_cycle, solve_lrs_priority_route)
from icir_cleanroom.gas_mapping.planning.lrs_tsp import LrsPoint, solve_lrs_tsp


def hrs_cell(variable, row, col, reward):
    return HrsCell(
        variable=variable, row=row, col=col, x=float(col), y=float(row),
        reward=reward, mean=reward, variance=0.1)


def test_held_karp_paths_preserve_order_and_length():
    cells = [hrs_cell(0, 0, 1, 0.1), hrs_cell(1, 1, 1, 0.2),
             hrs_cell(2, 1, 0, 0.3)]

    free_order, free_length = solve_held_karp_free_end_path(cells, (0.0, 0.0))
    open_order, open_length = solve_held_karp_open_path(cells, (0.0, 0.0))

    assert [cell.variable for cell in free_order] == [2, 1, 0]
    assert math.isclose(free_length, 3.0)
    assert [cell.variable for cell in open_order] == [2, 0, 1]
    assert math.isclose(open_length, 2.0 + math.sqrt(2.0))


def test_hrs_reward_ordered_exact_selects_best_feasible_set():
    candidates = [
        hrs_cell(0, 0, 1, 0.9), hrs_cell(1, 1, 1, 0.8),
        hrs_cell(2, 1, 0, 0.1)]
    result = solve_hrs_p2(
        candidates, (0.0, 0.0), visit_count=2, speed=5.0,
        update_seconds=50.0, dwell_seconds=2.0)

    assert result.solver == 'HELD_KARP'
    assert math.isclose(result.reward, 1.7)
    assert {cell.variable for cell in result.cells} == {0, 1}


def test_lrs_priority_preserves_complete_coverage():
    points = [
        LrsRewardPoint(i, row, col, float(col), float(row), reward,
                       mandatory=(i == 2))
        for i, (row, col, reward) in enumerate([
            (0, 0, 0.1), (0, 1, 0.9), (1, 1, 0.8), (1, 0, 0.2)])]
    result = solve_lrs_priority_route(
        points, (0.0, 0.0), candidate_count=4, priority_count=2,
        length_ratio_limit=2.0)

    assert len(result.points) == 4
    assert {point.variable for point in result.points} == {0, 1, 2, 3}
    assert 2 in {point.variable for point in result.priority_points}


def test_history_adjustment_replaces_near_points_adds_far_points_and_keeps_base():
    base = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    original_base = list(base)
    nearby = (1.0, 0.0)
    distant = (4.0, 5.0)

    adjusted, replacements, additions = history_adjusted_cycle(
        base, [nearby, distant], replace_radius=2.0,
        start_xy=(0.0, 0.0))

    assert base == original_base
    assert replacements == [((0.0, 0.0), nearby, 1.0)]
    assert additions == [distant]
    assert set(adjusted) == {nearby, distant, (10.0, 0.0), (10.0, 10.0)}


def test_lrs_tsp_square_is_a_closed_unit_tour():
    points = [
        LrsPoint(0, 0, 0.0, 0.0), LrsPoint(0, 1, 1.0, 0.0),
        LrsPoint(1, 1, 1.0, 1.0), LrsPoint(1, 0, 0.0, 1.0)]
    result = solve_lrs_tsp(points, start_xy=(0.0, 0.0), time_limit=5.0)

    assert result.status == 'OPTIMAL'
    assert math.isclose(result.objective, 4.0)
    assert sorted(result.tour) == [0, 1, 2, 3]


def test_lrs_tsp_uses_complete_domain_costs_after_obstacle_filtering():
    cells = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 2), (3, 0),
    ]
    points = [
        LrsPoint(row, col, float(col), float(row))
        for row, col in cells]
    distances = [[
        math.hypot(first.x - second.x, first.y - second.y)
        for second in points] for first in points]

    result = solve_lrs_tsp(
        points, start_xy=(1.0, 1.0), time_limit=5.0,
        distance_matrix=distances)

    assert result.status == 'OPTIMAL'
    assert sorted(result.tour) == list(range(len(points)))
