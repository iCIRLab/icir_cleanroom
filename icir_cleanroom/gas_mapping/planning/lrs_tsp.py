"""MILP TSP solver for the low-resolution sampling mesh."""

from dataclasses import dataclass
import math
import time

import numpy as np
from mip import BINARY, CBC, Model, OptimizationStatus, minimize, xsum


@dataclass(frozen=True)
class LrsPoint:
    row: int
    col: int
    x: float
    y: float


@dataclass
class TspResult:
    tour: list
    status: str
    gap: float
    objective: float
    cuts: int


def _components(node_count, selected_edges):
    adjacency = [[] for _ in range(node_count)]
    for i, j in selected_edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    remaining = set(range(node_count))
    components = []
    while remaining:
        stack = [remaining.pop()]
        component = set(stack)
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components, adjacency


def _validated_distance_matrix(distance_matrix, point_count):
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.shape != (point_count, point_count):
        raise ValueError(
            'LRS distance matrix must have one row and column per point')
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError('LRS distance matrix must be finite and non-negative')
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-9):
        raise ValueError('LRS distance matrix must be symmetric')
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1.0e-9):
        raise ValueError('LRS distance matrix diagonal must be zero')
    if np.any(matrix[np.triu_indices(point_count, 1)] <= 0.0):
        raise ValueError('distinct LRS points must have positive distance')
    return matrix


def solve_lrs_tsp(points, start_xy=(-5.0, -1.0), time_limit=60.0,
                  distance_matrix=None, start_distances=None):
    """Solve P1 by adding violated subtour elimination constraints."""
    if len(points) < 3:
        raise ValueError('A closed TSP tour requires at least three LRS points')

    edges = []
    if distance_matrix is None:
        for i, first in enumerate(points):
            for j in range(i + 1, len(points)):
                second = points[j]
                if max(abs(first.row - second.row),
                       abs(first.col - second.col)) == 1:
                    cost = math.hypot(
                        first.x - second.x, first.y - second.y)
                    edges.append((i, j, cost))
    else:
        matrix = _validated_distance_matrix(distance_matrix, len(points))
        edges = [
            (i, j, float(matrix[i, j]))
            for i in range(len(points))
            for j in range(i + 1, len(points))]

    model = Model(solver_name=CBC)
    model.verbose = 0
    variables = {(i, j): model.add_var(var_type=BINARY) for i, j, _ in edges}
    model.objective = minimize(
        xsum(cost * variables[i, j] for i, j, cost in edges)
    )
    for node in range(len(points)):
        model += xsum(
            variable for edge, variable in variables.items() if node in edge
        ) == 2

    started = time.monotonic()
    cuts = 0
    final_status = OptimizationStatus.NO_SOLUTION_FOUND
    adjacency = None
    while True:
        remaining = max(0.01, time_limit - (time.monotonic() - started))
        model.max_seconds = remaining
        final_status = model.optimize()
        if final_status not in (
            OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE
        ):
            raise RuntimeError(f'TSP has no valid incumbent: {final_status.name}')
        selected = [edge for edge, var in variables.items() if var.x >= 0.5]
        components, adjacency = _components(len(points), selected)
        if len(components) == 1:
            break
        if time.monotonic() - started >= time_limit:
            raise RuntimeError('TSP time limit reached before a single tour was found')
        for component in components:
            internal = [
                variables[edge] for edge in variables
                if edge[0] in component and edge[1] in component
            ]
            model += xsum(internal) <= len(component) - 1
            cuts += 1

    if start_distances is None:
        entry_costs = [math.hypot(
            point.x - start_xy[0], point.y - start_xy[1])
            for point in points]
    else:
        entry_costs = np.asarray(start_distances, dtype=float)
        if (entry_costs.shape != (len(points),) or
                not np.all(np.isfinite(entry_costs)) or
                np.any(entry_costs < 0.0)):
            raise ValueError(
                'LRS start distances must be finite and non-negative')
    start = min(range(len(points)), key=lambda index: entry_costs[index])
    tour = [start]
    previous = None
    current = start
    while len(tour) < len(points):
        choices = sorted(n for n in adjacency[current] if n != previous)
        following = choices[0]
        if following == start and len(tour) < len(points):
            following = choices[-1]
        tour.append(following)
        previous, current = current, following

    return TspResult(
        tour=tour,
        status=final_status.name,
        gap=float(model.gap),
        objective=float(model.objective_value),
        cuts=cuts,
    )
