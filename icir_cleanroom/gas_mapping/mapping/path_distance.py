"""Obstacle-aware distances over the robot sampling domain."""

import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


class SamplingDistanceOracle:
    """Compute reusable shortest-path distances without cutting corners."""

    _MOVES = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    )

    def __init__(self, geometry, free_mask):
        geometry.validate()
        mask = np.asarray(free_mask, dtype=bool)
        expected = (geometry.height, geometry.width)
        if mask.shape != expected:
            raise ValueError(
                f'sampling mask shape {mask.shape} does not match '
                f'geometry {expected}')
        self.geometry = geometry
        self.free_mask = mask.copy()
        self._distances = {}
        self._current_origin = None
        self._current_distances = None
        self._nearest_free_indices = None

    def _free_cell(self, point):
        row, col = self.geometry.world_to_cell(*point)
        if (not self.geometry.contains_cell(row, col) or
                not self.free_mask[row, col]):
            raise ValueError(
                f'point ({float(point[0]):.3f}, '
                f'{float(point[1]):.3f}) is outside the sampling domain')
        return int(row), int(col)

    def _nearest_free_cell(self, point):
        """Resolve a real robot pose to its nearest free cell, with no
        distance limit: the robot is really standing there, so a distance
        calculation from its current position must never be rejected just
        because that raw pose sits inside an obstacle-clearance buffer (or,
        in a pathological case, off the tracked grid entirely)."""
        row, col = self.geometry.world_to_cell(*point)
        row = min(max(row, 0), self.geometry.height - 1)
        col = min(max(col, 0), self.geometry.width - 1)
        if self.free_mask[row, col]:
            return int(row), int(col)
        if self._nearest_free_indices is None:
            _, self._nearest_free_indices = distance_transform_edt(
                ~self.free_mask, return_indices=True)
        nearest_row, nearest_col = self._nearest_free_indices[:, row, col]
        return int(nearest_row), int(nearest_col)

    def _cell_distances(self, start):
        cached = self._distances.get(start)
        if cached is not None:
            return cached
        distances = self._compute_cell_distances(start)
        self._distances[start] = distances
        return distances

    def _compute_cell_distances(self, start):
        """One Dijkstra traversal from start to all reachable map cells."""
        distances = np.full(self.free_mask.shape, np.inf, dtype=float)
        distances[start] = 0.0
        queue = [(0.0, start[0], start[1])]
        while queue:
            cost, row, col = heapq.heappop(queue)
            if cost > distances[row, col] + 1.0e-12:
                continue
            for delta_row, delta_col, multiplier in self._MOVES:
                next_row = row + delta_row
                next_col = col + delta_col
                if (next_row < 0 or next_col < 0 or
                        next_row >= self.geometry.height or
                        next_col >= self.geometry.width or
                        not self.free_mask[next_row, next_col]):
                    continue
                if delta_row and delta_col and (
                        not self.free_mask[row + delta_row, col] or
                        not self.free_mask[row, col + delta_col]):
                    continue
                candidate = cost + multiplier * self.geometry.resolution
                if candidate + 1.0e-12 < distances[next_row, next_col]:
                    distances[next_row, next_col] = candidate
                    heapq.heappush(
                        queue, (candidate, next_row, next_col))
        return distances

    def distances_from(self, first, targets):
        """Batch HRS distances with a bounded, one-origin distance-field cache.

        Uses the same grid, 8-connected edge costs and corner restrictions as
        distance(). A new sampling map creates a new oracle in the controller.
        LRS pairwise caches are intentionally left unchanged.
        """
        targets = tuple((float(x), float(y)) for x, y in targets)
        if not targets:
            return ()
        first = (float(first[0]), float(first[1]))
        origin = self._nearest_free_cell(first)
        cells = [self._free_cell(point) for point in targets]
        if self._current_origin != origin:
            self._current_distances = self._compute_cell_distances(origin)
            self._current_origin = origin
        result = []
        for point, cell in zip(targets, cells):
            value = (math.dist(first, point) if cell == origin
                     else float(self._current_distances[cell]))
            if not math.isfinite(value):
                raise ValueError(
                    f'no sampling-domain path connects {first} and {point}')
            result.append(value)
        return tuple(result)

    def _direct_grid_cost(self, first, second):
        """Return an obstacle-free optimal octile path, when available."""
        row, col = first
        target_row, target_col = second
        delta_row = abs(target_row - row)
        delta_col = abs(target_col - col)
        step_row = 1 if row < target_row else -1
        step_col = 1 if col < target_col else -1
        error = delta_col - delta_row
        cost = 0.0
        while (row, col) != (target_row, target_col):
            previous_row, previous_col = row, col
            twice_error = 2 * error
            if twice_error > -delta_row:
                error -= delta_row
                col += step_col
            if twice_error < delta_col:
                error += delta_col
                row += step_row
            if not self.free_mask[row, col]:
                return None
            moved_row = row - previous_row
            moved_col = col - previous_col
            if moved_row and moved_col:
                if (not self.free_mask[row, previous_col] or
                        not self.free_mask[previous_row, col]):
                    return None
                cost += math.sqrt(2.0) * self.geometry.resolution
            else:
                cost += self.geometry.resolution
        return cost

    def distance(self, first, second):
        first = (float(first[0]), float(first[1]))
        second = (float(second[0]), float(second[1]))
        first_cell = self._free_cell(first)
        second_cell = self._free_cell(second)
        if first_cell == second_cell:
            return math.hypot(
                first[0] - second[0], first[1] - second[1])
        source_cell, target_cell = sorted((first_cell, second_cell))
        cell_distance = self._direct_grid_cost(source_cell, target_cell)
        if cell_distance is None:
            cell_distance = self._cell_distances(source_cell)[target_cell]
        if not math.isfinite(float(cell_distance)):
            raise ValueError(
                f'no sampling-domain path connects {first} and {second}')
        return float(cell_distance)

    def matrix(self, points):
        coordinates = [
            (float(point.x), float(point.y)) for point in points]
        matrix = np.zeros((len(coordinates), len(coordinates)), dtype=float)
        for first in range(len(coordinates)):
            for second in range(first + 1, len(coordinates)):
                value = self.distance(
                    coordinates[first], coordinates[second])
                matrix[first, second] = value
                matrix[second, first] = value
        return matrix


__all__ = ['SamplingDistanceOracle']
