"""Sparse isotropic eight-neighbor GMRF estimator."""

import math
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg

from .grid_geometry import GridGeometry


class GmrfGrid:
    def __init__(self, map_msg, smoothness=1.0, background_precision=1.0,
                 background_mean=0.0, observation_variance_scale=0.01,
                 observation_variance_floor=0.001,
                 cardinal_weight=0.75, diagonal_weight=0.25):
        self.geometry = GridGeometry.from_message(map_msg)
        self.width = self.geometry.width
        self.height = self.geometry.height
        self.resolution = self.geometry.resolution
        self.origin_x = self.geometry.origin_x
        self.origin_y = self.geometry.origin_y
        self.field_mask = (
            np.asarray(map_msg.data).reshape(self.height, self.width) == 0)
        # Compatibility alias while callers migrate from the old navigation-
        # free interpretation to the gas-field interpretation.
        self.free = self.field_mask
        self.cell_to_var = np.full((self.height, self.width), -1, dtype=np.int32)
        self.cell_to_var[self.free] = np.arange(np.count_nonzero(self.free))
        self.var_cells = np.argwhere(self.free)
        self.observation_variance_scale = float(observation_variance_scale)
        self.observation_variance_floor = float(observation_variance_floor)
        self.cardinal_weight = float(cardinal_weight)
        self.diagonal_weight = float(diagonal_weight)
        self.background_precision = float(background_precision)
        self.background_mean = float(background_mean)
        if smoothness <= 0.0 or background_precision <= 0.0:
            raise ValueError('GMRF precisions must be positive')
        if (self.observation_variance_scale <= 0.0 or
                self.observation_variance_floor <= 0.0):
            raise ValueError('observation variance parameters must be positive')
        if self.cardinal_weight <= 0.0 or self.diagonal_weight < 0.0:
            raise ValueError('neighbor weights must be non-negative')
        self.obs_weight = np.zeros(len(self.var_cells))
        self.obs_rhs = np.zeros(len(self.var_cells))
        self.solution = np.full(len(self.var_cells), float(background_mean))
        self.variance = np.full(len(self.var_cells), 1.0 / background_precision)

        rows, cols, data = [], [], []
        degree = np.zeros(len(self.var_cells))
        graph_edges = []
        neighbor_edges = (
            (0, 1, self.cardinal_weight),
            (1, 0, self.cardinal_weight),
            (1, 1, self.diagonal_weight),
            (1, -1, self.diagonal_weight),
        )
        for row, col in self.var_cells:
            source = self.cell_to_var[row, col]
            for dr, dc, neighbor_weight in neighbor_edges:
                if neighbor_weight == 0.0:
                    continue
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.height and 0 <= nc < self.width and
                        self.free[nr, nc]):
                    target = self.cell_to_var[nr, nc]
                    edge_precision = smoothness * neighbor_weight
                    graph_edges.append((source, target, edge_precision))
                    rows.extend((source, target))
                    cols.extend((target, source))
                    data.extend((-edge_precision, -edge_precision))
                    degree[source] += edge_precision
                    degree[target] += edge_precision
        indices = np.arange(len(self.var_cells))
        rows.extend(indices.tolist())
        cols.extend(indices.tolist())
        data.extend((degree + background_precision).tolist())
        self.base = coo_matrix(
            (data, (rows, cols)), shape=(len(indices), len(indices))
        ).tocsr()
        self.background_rhs = np.full(
            len(indices), background_precision * background_mean
        )
        self.edge_source = np.asarray(
            [node for first, second, _ in graph_edges for node in (first, second)],
            dtype=np.int32)
        self.edge_target = np.asarray(
            [node for first, second, _ in graph_edges for node in (second, first)],
            dtype=np.int32)
        self.edge_precision = np.asarray(
            [value for _, _, value in graph_edges for _ in range(2)],
            dtype=float)
        self.reverse_edge = np.arange(len(self.edge_source), dtype=np.int32) ^ 1
        self.message_precision = np.zeros(len(self.edge_source), dtype=float)
        self.message_information = np.zeros(len(self.edge_source), dtype=float)

    def _nearest_variable(self, x, y):
        row, col = self.geometry.world_to_cell(x, y)
        if 0 <= row < self.height and 0 <= col < self.width:
            variable = self.cell_to_var[row, col]
            if variable >= 0:
                return int(variable)
        centers_x = self.origin_x + (self.var_cells[:, 1] + 0.5) * self.resolution
        centers_y = self.origin_y + (self.var_cells[:, 0] + 0.5) * self.resolution
        return int(np.argmin((centers_x - x) ** 2 + (centers_y - y) ** 2))

    def nearest_variable(self, x, y):
        """Return the variable corresponding to an actual world pose."""
        return self._nearest_variable(float(x), float(y))

    def set_observation(self, x, y, value):
        """Replace the observation at the nearest cell with the latest value."""
        variable = self._nearest_variable(x, y)
        value = min(1.0, max(0.0, float(value)))
        variance = max(
            self.observation_variance_floor,
            self.observation_variance_scale * abs(value))
        precision = 1.0 / variance
        self.obs_weight[variable] = precision
        self.obs_rhs[variable] = precision * value
        return tuple(self.var_cells[variable])

    def add_observation(self, x, y, value):
        """Compatibility alias; observations now use latest-value semantics."""
        return self.set_observation(x, y, value)

    def reset_observations(self):
        """Start an independent mapping episode from the background prior."""
        self.obs_weight.fill(0.0)
        self.obs_rhs.fill(0.0)
        self.solution.fill(self.background_mean)
        self.variance.fill(1.0 / self.background_precision)
        self.reset_gabp_messages()

    def variable_at(self, row, col):
        if 0 <= row < self.height and 0 <= col < self.width:
            variable = int(self.cell_to_var[row, col])
            if variable >= 0:
                return variable
        return None

    def cell_center(self, variable):
        row, col = self.var_cells[int(variable)]
        return self.geometry.cell_center(row, col)

    def neighbor_variables(self, variable):
        """Return all valid 8-connected neighbors in row/column order."""
        row, col = self.var_cells[int(variable)]
        neighbors = []
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                neighbor = self.variable_at(
                    int(row + row_offset), int(col + col_offset))
                if neighbor is not None:
                    neighbors.append(neighbor)
        return neighbors

    def reset_gabp_messages(self):
        self.message_precision.fill(0.0)
        self.message_information.fill(0.0)

    def solve_gabp(self, max_iterations=500, tolerance=1.0e-6, damping=0.5):
        if not 0.0 < damping <= 1.0:
            raise ValueError('GaBP damping must be in (0, 1]')
        # Pairwise terms already contribute to the sparse matrix diagonal;
        # GaBP unary factors contain only background and observations.
        background_precision = self.base.diagonal() - np.bincount(
            self.edge_source, weights=self.edge_precision,
            minlength=len(self.var_cells))
        unary_precision = background_precision + self.obs_weight
        unary_information = self.background_rhs + self.obs_rhs

        residual = math.inf
        for iteration in range(1, int(max_iterations) + 1):
            incoming_precision = np.bincount(
                self.edge_target, weights=self.message_precision,
                minlength=len(self.var_cells))
            incoming_information = np.bincount(
                self.edge_target, weights=self.message_information,
                minlength=len(self.var_cells))
            total_precision = unary_precision + incoming_precision
            total_information = unary_information + incoming_information

            cavity_precision = (
                total_precision[self.edge_source] -
                self.message_precision[self.reverse_edge])
            cavity_information = (
                total_information[self.edge_source] -
                self.message_information[self.reverse_edge])
            denominator = self.edge_precision + cavity_precision
            proposed_precision = (
                self.edge_precision * cavity_precision / denominator)
            proposed_information = (
                self.edge_precision * cavity_information / denominator)
            updated_precision = (
                (1.0 - damping) * self.message_precision +
                damping * proposed_precision)
            updated_information = (
                (1.0 - damping) * self.message_information +
                damping * proposed_information)
            residual = max(
                float(np.max(np.abs(
                    updated_precision - self.message_precision))),
                float(np.max(np.abs(
                    updated_information - self.message_information))),
            )
            self.message_precision = updated_precision
            self.message_information = updated_information
            if not (np.all(np.isfinite(self.message_precision)) and
                    np.all(np.isfinite(self.message_information))):
                return False, iteration, math.inf
            if residual <= tolerance:
                break
        else:
            return False, int(max_iterations), residual

        incoming_precision = np.bincount(
            self.edge_target, weights=self.message_precision,
            minlength=len(self.var_cells))
        incoming_information = np.bincount(
            self.edge_target, weights=self.message_information,
            minlength=len(self.var_cells))
        marginal_precision = unary_precision + incoming_precision
        marginal_information = unary_information + incoming_information
        candidate = marginal_information / marginal_precision
        variance = 1.0 / marginal_precision
        if not (np.all(np.isfinite(candidate)) and
                np.all(np.isfinite(variance)) and np.all(variance > 0.0)):
            return False, iteration, math.inf
        self.solution = np.clip(candidate, 0.0, 1.0)
        self.variance = variance
        return True, iteration, residual

    def cg_reference(self, max_iterations=2000, tolerance=1.0e-6):
        matrix = self.base + diags(self.obs_weight)
        rhs = self.background_rhs + self.obs_rhs
        preconditioner = diags(1.0 / matrix.diagonal())
        candidate, info = cg(
            matrix, rhs, x0=self.solution, tol=tolerance,
            maxiter=max_iterations, M=preconditioner)
        if info != 0 or not np.all(np.isfinite(candidate)):
            return None, int(info)
        return np.clip(candidate, 0.0, 1.0), 0

    def solve(self, max_iterations=2000, tolerance=1.0e-6):
        matrix = self.base + diags(self.obs_weight)
        rhs = self.background_rhs + self.obs_rhs
        preconditioner = diags(1.0 / matrix.diagonal())
        candidate, info = cg(
            matrix, rhs, x0=self.solution, tol=tolerance,
            maxiter=max_iterations, M=preconditioner
        )
        if info != 0 or not np.all(np.isfinite(candidate)):
            return False, int(info)
        self.solution = np.clip(candidate, 0.0, 1.0)
        return True, 0

    def occupancy_data(self):
        output = np.full((self.height, self.width), -1, dtype=np.int8)
        output[self.free] = np.rint(100.0 * self.solution).astype(np.int8)
        return output.ravel().tolist()
