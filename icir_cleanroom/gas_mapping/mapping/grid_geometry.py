"""Grid/world coordinate transforms shared by mapping and planning."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GridGeometry:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float = 0.0

    @classmethod
    def from_message(cls, message):
        orientation = getattr(message.info.origin, 'orientation', None)
        if orientation is None:
            yaw = 0.0
        else:
            siny = 2.0 * (
                float(getattr(orientation, 'w', 1.0)) *
                float(getattr(orientation, 'z', 0.0)) +
                float(getattr(orientation, 'x', 0.0)) *
                float(getattr(orientation, 'y', 0.0)))
            cosy = 1.0 - 2.0 * (
                float(getattr(orientation, 'y', 0.0)) ** 2 +
                float(getattr(orientation, 'z', 0.0)) ** 2)
            yaw = math.atan2(siny, cosy)
        geometry = cls(
            int(message.info.width), int(message.info.height),
            float(message.info.resolution),
            float(message.info.origin.position.x),
            float(message.info.origin.position.y), yaw)
        geometry.validate()
        return geometry

    def validate(self):
        values = (
            self.resolution, self.origin_x, self.origin_y, self.origin_yaw)
        if self.width <= 0 or self.height <= 0:
            raise ValueError('grid width and height must be positive')
        if not all(math.isfinite(value) for value in values):
            raise ValueError('grid geometry values must be finite')
        if self.resolution <= 0.0:
            raise ValueError('grid resolution must be positive')

    def world_to_cell(self, x, y):
        local_x, local_y = self._world_to_local(x, y)
        col = int(math.floor(local_x / self.resolution))
        row = int(math.floor(local_y / self.resolution))
        return row, col

    def _world_to_local(self, x, y):
        dx = float(x) - self.origin_x
        dy = float(y) - self.origin_y
        cosine = math.cos(self.origin_yaw)
        sine = math.sin(self.origin_yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        return local_x, local_y

    def cell_center(self, row, col):
        local_x = (float(col) + 0.5) * self.resolution
        local_y = (float(row) + 0.5) * self.resolution
        cosine = math.cos(self.origin_yaw)
        sine = math.sin(self.origin_yaw)
        return (
            self.origin_x + cosine * local_x - sine * local_y,
            self.origin_y + sine * local_x + cosine * local_y,
        )

    def contains_cell(self, row, col):
        return 0 <= int(row) < self.height and 0 <= int(col) < self.width

    def contains_world(self, x, y):
        return self.contains_cell(*self.world_to_cell(x, y))


__all__ = ['GridGeometry']
