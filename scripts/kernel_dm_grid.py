"""
Kernel DM (Lilienthal & Duckett, 2004) 기반 가스 농도 그리드맵.
로봇 위치 중심의 가우시안 커널로 측정값을 격자에 누적하고,
누적된 지도에서 argmax 셀을 소스 위치 추정치로 제공한다.
"""
import math
import numpy as np


class KernelDMGrid:
    def __init__(self, origin_x, origin_y, resolution, width, height,
                 sigma=1.5, cutoff_radius=None, w_min=1.0):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.resolution = resolution
        self.width = width
        self.height = height
        self.sigma = sigma
        self.cutoff_radius = cutoff_radius if cutoff_radius is not None else 3.0 * sigma
        self.w_min = w_min

        self.value_sum = np.zeros((height, width), dtype=np.float64)
        self.weight_sum = np.zeros((height, width), dtype=np.float64)

    def reset(self):
        # 새 가스 소스 에피소드 시작 시 이전 누적 히트맵을 비운다
        self.value_sum.fill(0.0)
        self.weight_sum.fill(0.0)

    def _world_to_cell(self, x, y):
        i = int((x - self.origin_x) / self.resolution)
        j = int((y - self.origin_y) / self.resolution)
        return i, j

    def _cell_to_world(self, i, j):
        x = self.origin_x + (i + 0.5) * self.resolution
        y = self.origin_y + (j + 0.5) * self.resolution
        return x, y

    def update(self, robot_x, robot_y, concentration):
        center_i, center_j = self._world_to_cell(robot_x, robot_y)
        offset = int(self.cutoff_radius / self.resolution)

        i_min, i_max = max(0, center_i - offset), min(self.width, center_i + offset + 1)
        j_min, j_max = max(0, center_j - offset), min(self.height, center_j + offset + 1)
        if i_min >= i_max or j_min >= j_max:
            return

        ii, jj = np.meshgrid(np.arange(i_min, i_max), np.arange(j_min, j_max))
        world_x = self.origin_x + (ii + 0.5) * self.resolution
        world_y = self.origin_y + (jj + 0.5) * self.resolution
        dist = np.hypot(world_x - robot_x, world_y - robot_y)

        mask = dist <= self.cutoff_radius
        weight = np.where(mask, np.exp(-(dist ** 2) / (2.0 * self.sigma ** 2)), 0.0)

        self.weight_sum[j_min:j_max, i_min:i_max] += weight
        self.value_sum[j_min:j_max, i_min:i_max] += concentration * weight

    def get_cell_value(self, x, y):
        i, j = self._world_to_cell(x, y)
        if not (0 <= i < self.width and 0 <= j < self.height):
            return None
        if self.weight_sum[j, i] < self.w_min:
            return None
        return self.value_sum[j, i] / self.weight_sum[j, i]

    def get_argmax_position(self):
        valid = self.weight_sum >= self.w_min
        if not np.any(valid):
            return None

        with np.errstate(invalid='ignore', divide='ignore'):
            mean_grid = np.where(valid, self.value_sum / self.weight_sum, -np.inf)

        j, i = np.unravel_index(np.argmax(mean_grid), mean_grid.shape)
        return self._cell_to_world(i, j)

    def to_occupancy_grid_data(self, normalize_max=500.0):
        valid = self.weight_sum >= self.w_min
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_grid = np.where(valid, self.value_sum / self.weight_sum, 0.0)

        scaled = np.clip(mean_grid / normalize_max * 100.0, 0, 100).astype(np.int8)
        data = np.where(valid, scaled, -1).astype(np.int8)
        return data.flatten().tolist()
