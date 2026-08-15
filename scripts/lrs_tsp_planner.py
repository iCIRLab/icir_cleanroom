"""MILP TSP solver for the low-resolution sampling mesh."""

from dataclasses import dataclass
import math
import time

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


def solve_lrs_tsp(points, start_xy=(-5.0, -1.0), time_limit=60.0):
    """Solve P1 by adding violated subtour elimination constraints."""
    if len(points) < 3:
        raise ValueError('A closed TSP tour requires at least three LRS points')

    edges = []
    for i, first in enumerate(points):
        for j in range(i + 1, len(points)):
            second = points[j]
            if max(abs(first.row - second.row), abs(first.col - second.col)) == 1:
                cost = math.hypot(first.x - second.x, first.y - second.y)
                edges.append((i, j, cost))

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

    start = min(
        range(len(points)),
        key=lambda index: math.hypot(
            points[index].x - start_xy[0], points[index].y - start_xy[1]
        ),
    )
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
