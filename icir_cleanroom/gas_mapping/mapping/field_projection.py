"""ROS-independent projection and display transforms for GMRF fields."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class ProjectedField:
    values: np.ndarray
    free_mask: np.ndarray
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float


def project_field(gmrf, variable_values, display_resolution):
    """Interpolate variable values onto the requested display grid."""
    model_resolution = float(gmrf.resolution)
    display_resolution = float(display_resolution)
    if display_resolution <= 0.0:
        raise ValueError('estimate_resolution must be positive')
    field = np.zeros((gmrf.height, gmrf.width), dtype=float)
    field[gmrf.free] = np.asarray(variable_values, dtype=float)
    if math.isclose(display_resolution, model_resolution):
        return ProjectedField(
            field, np.asarray(gmrf.free, dtype=bool).copy(), model_resolution,
            int(gmrf.width), int(gmrf.height), float(gmrf.origin_x),
            float(gmrf.origin_y))

    min_x = gmrf.origin_x + 0.5 * model_resolution
    min_y = gmrf.origin_y + 0.5 * model_resolution
    max_x = gmrf.origin_x + (gmrf.width - 0.5) * model_resolution
    max_y = gmrf.origin_y + (gmrf.height - 0.5) * model_resolution
    width = int(round((max_x - min_x) / display_resolution)) + 1
    height = int(round((max_y - min_y) / display_resolution)) + 1
    x_world = min_x + np.arange(width) * display_resolution
    y_world = min_y + np.arange(height) * display_resolution
    grid_x, grid_y = np.meshgrid(
        (x_world - min_x) / model_resolution,
        (y_world - min_y) / model_resolution)
    interpolated = map_coordinates(
        field, [grid_y, grid_x], order=1, mode='nearest')
    return ProjectedField(
        interpolated, np.ones((height, width), dtype=bool),
        display_resolution, width, height,
        min_x - 0.5 * display_resolution,
        min_y - 0.5 * display_resolution)


def logarithmic_display(values, log_min):
    """Map linear concentrations to a bounded logarithmic display field."""
    log_min = float(log_min)
    if log_min >= 0.0:
        raise ValueError('log_concentration_min must be negative')
    log_values = np.log(np.clip(values, math.exp(log_min), 1.0))
    return np.clip((log_values - log_min) / -log_min, 0.0, 1.0)


__all__ = ['ProjectedField', 'logarithmic_display', 'project_field']
