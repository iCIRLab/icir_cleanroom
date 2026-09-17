"""Composable HRS selectors. Deliberately accepts no source truth or ROS node."""
from dataclasses import dataclass, replace
import math

from .hrs import HrsManager
from .hrs_termination import attempts_exhausted


@dataclass(frozen=True)
class Method:
    initial: str
    search: str


METHODS = {
    'M1': Method('lrs_max', 'dducb'),
    'M2': Method('lrs_max', 'spiral'),
    'M3': Method('dducb', 'dducb'),
    'M4': Method('dducb', 'spiral'),
    'M5': Method('lrs_centroid', 'dducb'),
    'M6': Method('lrs_centroid', 'spiral'),
    'M7': Method('ucb', 'ucb'),
}


def select_ucb(candidates, **context):
    scored = tuple(replace(c, score=c.ucb) for c in candidates)
    return scored, min(scored, key=lambda c: (
        -c.score, c.row, c.col, c.variable)) if scored else None


def select_dducb(candidates, *, current_xy, oracle, distance_weight, **context):
    return HrsManager.select_candidate(
        candidates, current_xy, oracle.distance, distance_weight,
        distances=oracle.distances_from(
            current_xy, ((cell.x, cell.y) for cell in candidates)))


def lrs_max(measurements):
    if not measurements:
        raise ValueError('no valid measurements in current LRS lap')
    m = max(measurements, key=lambda m: m.value)
    return m.pose.x, m.pose.y


def lrs_centroid(measurements):
    weights = [max(0., m.value) for m in measurements]
    total = sum(weights)
    if total <= 0:
        raise ValueError('current LRS centroid has zero total concentration')
    return (sum(m.pose.x*w for m, w in zip(measurements, weights))/total,
            sum(m.pose.y*w for m, w in zip(measurements, weights))/total)


SCORE_SELECTORS = {'ucb': select_ucb, 'dducb': select_dducb}
POINT_SELECTORS = {'lrs_max': lrs_max, 'lrs_centroid': lrs_centroid}


class SquareSpiral:
    """Finite east/north/west/south spiral; all coordinates are field cells."""
    def __init__(self, step=1, epsilon=0.):
        self.step = step
        self.epsilon = epsilon
        self.center = None
        self.value = None
        self.visited = set()
        self.cursor = iter(())

    def recenter(self, cell, value, bounds):
        self.center, self.value = cell, value
        self.cursor = self._cells(cell, bounds)

    def _cells(self, center, bounds):
        r, c = center
        min_r, max_r, min_c, max_c = bounds
        radius = max(abs(r-min_r), abs(r-max_r), abs(c-min_c), abs(c-max_c))
        rings = math.ceil(radius/self.step)
        length = 1
        # Exactly 2*rings complete legs pairs cover the surrounding square.
        directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
        direction = 0
        for _ in range(2*rings+1):
            for _ in range(2):
                dr, dc = directions[direction % 4]
                for _ in range(length):
                    r += dr*self.step
                    c += dc*self.step
                    yield r, c
                direction += 1
            length += 1

    def observe(self, cell, value, bounds):
        self.visited.add(cell)
        if self.center is None or value > self.value + self.epsilon:
            self.recenter(cell, value, bounds)
            return True
        return False

    def select(self, candidates, **context):
        by_cell = {(c.row, c.col): c for c in candidates}
        for cell in self.cursor:
            if cell in by_cell and cell not in self.visited:
                return candidates, by_cell[cell]
        return candidates, None


class HrsSession:
    """Fresh per-alert strategy state, with an immutable current-lap snapshot."""
    def __init__(self, method='M3', measurements=(), step=1, epsilon=0.,
                 patience=3, min_relative_improvement=0.05):
        self.method_id = method
        self.method = METHODS[method]
        self.stage = 'INITIAL'
        self.measurements = tuple(m for m in measurements if all(
            math.isfinite(v) for v in (m.pose.x, m.pose.y, m.value)))
        self.spiral = SquareSpiral(step, epsilon)
        self.selectors = dict(SCORE_SELECTORS, spiral=self.spiral.select)
        self.initial_target = None
        self.requested_xy = None
        self.attempt_count = 0
        self.search_iterations = 0
        self.patience = patience
        self.min_relative_improvement = min_relative_improvement
        self.best_value = None
        self.no_improvement_streak = 0

    def select(self, candidates, *, gmrf, current_xy, oracle, distance_weight,
               position_valid=None):
        candidates = tuple(candidates)
        strategy = self.method.initial if self.stage == 'INITIAL' else self.method.search
        if strategy in POINT_SELECTORS:
            requested = POINT_SELECTORS[strategy](self.measurements)
            # Snap the raw LRS point to its own GMRF cell center first, so
            # the computation itself is in cell units, not a continuous
            # LRS pose; this is then also the nearest candidate's center.
            row, col = gmrf.geometry.world_to_cell(*requested)
            self.requested_xy = gmrf.geometry.cell_center(row, col)
            if not candidates:
                return candidates, None
            x, y = self.requested_xy
            cell = min(candidates, key=lambda c: (
                math.hypot(c.x-x, c.y-y), c.row, c.col, c.variable))
            scored, selected = candidates, cell
        else:
            scored, selected = self.selectors[strategy](
                candidates, current_xy=current_xy, oracle=oracle,
                distance_weight=distance_weight)
        if selected is not None:
            self.attempt_count += 1
            if self.stage == 'SEARCH':
                self.search_iterations += 1
            else:
                self.initial_target = selected
        return scored, selected

    def is_relative_improvement(self, value):
        if self.best_value is None:
            return True
        if self.best_value <= 0.0:
            return value > self.best_value
        return value - self.best_value >= self.min_relative_improvement * self.best_value

    def stalled(self):
        """True once SEARCH has gone `patience` attempts without a relative gain."""
        return attempts_exhausted(self.no_improvement_streak, self.patience)

    def updated(self, variable, value, gmrf):
        """Called only after a valid observation's GMRF update succeeds."""
        initial = self.stage == 'INITIAL'
        self.stage = 'SEARCH'
        if self.is_relative_improvement(value):
            self.best_value = value
            self.no_improvement_streak = 0
        elif not initial:
            self.no_improvement_streak += 1
        bounds = (0, gmrf.height-1, 0, gmrf.width-1)
        cell = tuple(int(v) for v in gmrf.var_cells[variable])
        if initial and self.method.search == 'spiral':
            target = self.initial_target
            self.spiral.recenter((target.row, target.col), value, bounds)
            self.spiral.visited.add((target.row, target.col))
            self.spiral.visited.add(cell)
            return True
        return self.method.search == 'spiral' and self.spiral.observe(cell, value, bounds)
