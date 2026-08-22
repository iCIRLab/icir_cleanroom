"""Exact P2 planners for high-resolution gas sampling."""

from dataclasses import dataclass
from itertools import combinations
import math
import time

from mip import BINARY, CBC, Model, OptimizationStatus, minimize, xsum

from .exact_path import (
    solve_held_karp_free_end_path, solve_held_karp_open_path)


REWARD_ORDERED_EXACT = 'reward_ordered_exact'
PAPER_EXHAUSTIVE_MILP = 'paper_exhaustive_milp'
PLANNER_MODES = (REWARD_ORDERED_EXACT, PAPER_EXHAUSTIVE_MILP)


@dataclass(frozen=True)
class HrsCell:
    variable: int
    row: int
    col: int
    x: float
    y: float
    reward: float
    mean: float
    variance: float


@dataclass
class HrsRoute:
    cells: list
    reward: float
    length: float
    expected_seconds: float
    status: str
    gap: float
    cuts: int
    visit_count: int
    total_combinations: int
    evaluated_routes: int
    pruned_combinations: int
    feasible_combinations: int
    solver: str
    planning_seconds: float


def _farthest_cell_index(cells, start_xy):
    return max(
        range(len(cells)),
        key=lambda index: (
            math.hypot(cells[index].x - start_xy[0],
                       cells[index].y - start_xy[1]),
            -cells[index].row, -cells[index].col),
    )


def _components(node_count, selected_edges):
    adjacency = [[] for _ in range(node_count)]
    for first, second in selected_edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    remaining = set(range(node_count))
    components = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components, adjacency


def _ordered_open_path(adjacency, endpoint):
    path = [0]
    previous = -1
    current = 0
    while current != endpoint:
        next_nodes = [node for node in adjacency[current] if node != previous]
        if len(next_nodes) != 1:
            return None
        following = next_nodes[0]
        if following in path:
            return None
        path.append(following)
        previous, current = current, following
    if len(path) != len(adjacency):
        return None
    return path


def solve_modified_tsp(cells, start_xy, time_limit=5.0):
    """Solve the paper's equation (3) MILP for one cell combination."""
    if not cells:
        return None
    node_xy = [tuple(start_xy)] + [(cell.x, cell.y) for cell in cells]
    endpoint = _farthest_cell_index(cells, start_xy) + 1
    if len(cells) == 1:
        length = math.hypot(
            cells[0].x - start_xy[0], cells[0].y - start_xy[1])
        return [cells[0]], length, 'DIRECT', 0.0, 0

    edges = []
    for first in range(len(node_xy)):
        for second in range(first + 1, len(node_xy)):
            cost = math.hypot(
                node_xy[first][0] - node_xy[second][0],
                node_xy[first][1] - node_xy[second][1])
            edges.append((first, second, cost))

    model = Model(solver_name=CBC)
    model.verbose = 0
    variables = {
        (first, second): model.add_var(var_type=BINARY)
        for first, second, _ in edges
    }
    model.objective = minimize(xsum(
        cost * variables[first, second]
        for first, second, cost in edges))
    for node in range(len(node_xy)):
        required_degree = 1 if node in (0, endpoint) else 2
        model += xsum(
            variable for edge, variable in variables.items() if node in edge
        ) == required_degree

    started = time.monotonic()
    cuts = 0
    final_status = OptimizationStatus.NO_SOLUTION_FOUND
    adjacency = None
    while True:
        remaining = float(time_limit) - (time.monotonic() - started)
        if remaining <= 0.0:
            return None
        model.max_seconds = max(0.01, remaining)
        final_status = model.optimize()
        if final_status not in (
                OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        selected = [edge for edge, variable in variables.items()
                    if variable.x is not None and variable.x >= 0.5]
        components, adjacency = _components(len(node_xy), selected)
        if len(components) == 1:
            break
        for component in components:
            internal = [
                variables[edge] for edge in variables
                if edge[0] in component and edge[1] in component
            ]
            model += xsum(internal) <= len(component) - 1
            cuts += 1

    order = _ordered_open_path(adjacency, endpoint)
    if order is None:
        return None
    ordered_cells = [cells[node - 1] for node in order[1:]]
    return (
        ordered_cells, float(model.objective_value), final_status.name,
        float(model.gap), cuts)


def _mst_lower_bound(cells, start_xy):
    """MST weight is a safe lower bound for any path visiting all nodes."""
    coordinates = [tuple(start_xy)] + [(cell.x, cell.y) for cell in cells]
    remaining = set(range(1, len(coordinates)))
    minimum_edge = {
        node: math.hypot(
            coordinates[node][0] - coordinates[0][0],
            coordinates[node][1] - coordinates[0][1])
        for node in remaining
    }
    total = 0.0
    while remaining:
        node = min(remaining, key=lambda item: (minimum_edge[item], item))
        total += minimum_edge[node]
        remaining.remove(node)
        for other in remaining:
            edge = math.hypot(
                coordinates[node][0] - coordinates[other][0],
                coordinates[node][1] - coordinates[other][1])
            minimum_edge[other] = min(minimum_edge[other], edge)
    return total


def _route_result(ordered, reward, length, speed, dwell_seconds,
                  visit_count, total, evaluated, pruned, feasible,
                  solver, planning_seconds, status='OPTIMAL', gap=0.0,
                  cuts=0):
    return HrsRoute(
        cells=ordered,
        reward=reward,
        length=length,
        expected_seconds=length / speed + visit_count * dwell_seconds,
        status=status,
        gap=gap,
        cuts=cuts,
        visit_count=visit_count,
        total_combinations=total,
        evaluated_routes=evaluated,
        pruned_combinations=pruned,
        feasible_combinations=feasible,
        solver=solver,
        planning_seconds=planning_seconds,
    )


def solve_peak_confirmation_route(cells, start_xy, speed=5.0,
                                  dwell_seconds=2.0):
    """Plan an exact shortest route through one peak's unmeasured neighbors."""
    if speed <= 0.0 or dwell_seconds < 0.0:
        raise ValueError('peak-confirmation timing parameters are invalid')
    started = time.monotonic()
    solved = solve_held_karp_free_end_path(cells, start_xy)
    if solved is None:
        return None
    ordered, length = solved
    reward = float(sum(cell.reward for cell in cells))
    return _route_result(
        ordered, reward, length, speed, dwell_seconds, len(cells), 1, 1,
        0, 1, 'HELD_KARP_PEAK', time.monotonic() - started)


def solve_reward_ordered_p2(candidates, start_xy, visit_count=10, speed=5.0,
                            update_seconds=50.0, dwell_seconds=2.0):
    """Find the exact maximum-reward feasible set without exhaustive TSPs."""
    started = time.monotonic()
    route_budget = speed * (update_seconds - visit_count * dwell_seconds)
    ranked = []
    for selected_tuple in combinations(candidates, visit_count):
        reward = float(sum(cell.reward for cell in selected_tuple))
        cell_key = tuple(sorted(
            (cell.row, cell.col) for cell in selected_tuple))
        ranked.append((-reward, cell_key, selected_tuple, reward))
    ranked.sort(key=lambda item: (item[0], item[1]))

    evaluated = 0
    pruned = 0
    for _, _, selected_tuple, reward in ranked:
        selected = list(selected_tuple)
        if _mst_lower_bound(selected, start_xy) > route_budget + 1.0e-9:
            pruned += 1
            continue
        evaluated += 1
        ordered, length = solve_held_karp_open_path(selected, start_xy)
        if length <= route_budget + 1.0e-9:
            return _route_result(
                ordered, reward, length, speed, dwell_seconds, visit_count,
                len(ranked), evaluated, pruned, 1, 'HELD_KARP',
                time.monotonic() - started)
    return None


def solve_paper_p2(candidates, start_xy, visit_count=10, speed=5.0,
                   update_seconds=50.0, dwell_seconds=2.0,
                   combination_time_limit=5.0):
    """Exhaustively reproduce the paper's M-choose-T MILP procedure."""
    started = time.monotonic()
    route_budget = speed * (update_seconds - visit_count * dwell_seconds)
    selected_sets = list(combinations(candidates, visit_count))
    evaluated = 0
    feasible = 0
    best = None
    best_key = None
    for selected_tuple in selected_sets:
        evaluated += 1
        solved = solve_modified_tsp(
            list(selected_tuple), start_xy, combination_time_limit)
        if solved is None:
            continue
        ordered, length, status, gap, cuts = solved
        if length > route_budget + 1.0e-9:
            continue
        feasible += 1
        reward = float(sum(cell.reward for cell in selected_tuple))
        cell_key = tuple(sorted((cell.row, cell.col) for cell in selected_tuple))
        selection_key = (-reward, length, cell_key)
        if best is None or selection_key < best_key:
            best_key = selection_key
            best = _route_result(
                ordered, reward, length, speed, dwell_seconds, visit_count,
                len(selected_sets), evaluated, 0, feasible, 'CBC', 0.0,
                status, gap, cuts)
    if best is not None:
        best.evaluated_routes = evaluated
        best.feasible_combinations = feasible
        best.planning_seconds = time.monotonic() - started
    return best


def solve_hrs_p2(candidates, start_xy, visit_count=10, speed=5.0,
                 update_seconds=50.0, dwell_seconds=2.0,
                 combination_time_limit=5.0,
                 planner_mode=REWARD_ORDERED_EXACT):
    """Dispatch to the optimized exact planner or paper reproduction mode."""
    if speed <= 0.0 or update_seconds <= 0.0 or dwell_seconds < 0.0:
        raise ValueError('P2 timing parameters are invalid')
    if planner_mode not in PLANNER_MODES:
        raise ValueError(
            f'Unknown HRS planner mode {planner_mode!r}; expected {PLANNER_MODES}')
    visit_count = min(int(visit_count), len(candidates))
    if visit_count <= 0:
        return None
    if speed * (update_seconds - visit_count * dwell_seconds) < 0.0:
        return None
    if planner_mode == REWARD_ORDERED_EXACT:
        return solve_reward_ordered_p2(
            candidates, start_xy, visit_count, speed, update_seconds,
            dwell_seconds)
    return solve_paper_p2(
        candidates, start_xy, visit_count, speed, update_seconds,
        dwell_seconds, combination_time_limit)
