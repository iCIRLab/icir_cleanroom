"""Sparse isotropic eight-neighbor GMRF MAP estimator."""

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg


class GmrfGrid:
    def __init__(self, map_msg, smoothness=1.0, background_precision=1.0,
                 background_mean=0.0, observation_variance_scale=0.01,
                 observation_variance_floor=0.001,
                 cardinal_weight=0.75, diagonal_weight=0.25):
        self.width = map_msg.info.width
        self.height = map_msg.info.height
        self.resolution = map_msg.info.resolution
        self.origin_x = map_msg.info.origin.position.x
        self.origin_y = map_msg.info.origin.position.y
        self.free = np.asarray(map_msg.data).reshape(self.height, self.width) == 0
        self.cell_to_var = np.full((self.height, self.width), -1, dtype=np.int32)
        self.cell_to_var[self.free] = np.arange(np.count_nonzero(self.free))
        self.var_cells = np.argwhere(self.free)
        self.observation_variance_scale = float(observation_variance_scale)
        self.observation_variance_floor = float(observation_variance_floor)
        self.cardinal_weight = float(cardinal_weight)
        self.diagonal_weight = float(diagonal_weight)
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

        rows, cols, data = [], [], []
        degree = np.zeros(len(self.var_cells))
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

    def _nearest_variable(self, x, y):
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        if 0 <= row < self.height and 0 <= col < self.width:
            variable = self.cell_to_var[row, col]
            if variable >= 0:
                return int(variable)
        centers_x = self.origin_x + (self.var_cells[:, 1] + 0.5) * self.resolution
        centers_y = self.origin_y + (self.var_cells[:, 0] + 0.5) * self.resolution
        return int(np.argmin((centers_x - x) ** 2 + (centers_y - y) ** 2))

    def add_observation(self, x, y, value):
        variable = self._nearest_variable(x, y)
        value = min(1.0, max(0.0, float(value)))
        variance = max(
            self.observation_variance_floor,
            self.observation_variance_scale * abs(value))
        precision = 1.0 / variance
        self.obs_weight[variable] += precision
        self.obs_rhs[variable] += precision * value
        return tuple(self.var_cells[variable])

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
