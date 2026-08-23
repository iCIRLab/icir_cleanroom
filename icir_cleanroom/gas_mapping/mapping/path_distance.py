"""Obstacle-aware distances over the robot sampling domain."""

import heapq
import math

import numpy as np


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

    def _free_cell(self, point):
        row, col = self.geometry.world_to_cell(*point)
        if (not self.geometry.contains_cell(row, col) or
                not self.free_mask[row, col]):
            raise ValueError(
                f'point ({float(point[0]):.3f}, '
                f'{float(point[1]):.3f}) is outside the sampling domain')
        return int(row), int(col)

    def _cell_distances(self, start):
        cached = self._distances.get(start)
        if cached is not None:
            return cached
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
        self._distances[start] = distances
        return distances

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
