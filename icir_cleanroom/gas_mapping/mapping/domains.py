"""Gas-field and robot-sampling domain construction."""

import math

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from .grid_geometry import GridGeometry


def field_geometry(min_x, max_x, min_y, max_y, resolution):
    values = [float(value) for value in (
        min_x, max_x, min_y, max_y, resolution)]
    min_x, max_x, min_y, max_y, resolution = values
    if not all(math.isfinite(value) for value in values):
        raise ValueError('gas field geometry values must be finite')
    if resolution <= 0.0:
        raise ValueError('gas field resolution must be positive')
    if max_x < min_x or max_y < min_y:
        raise ValueError('gas field maximums must not be below minimums')
    span_x = (max_x - min_x) / resolution
    span_y = (max_y - min_y) / resolution
    rounded_x = round(span_x)
    rounded_y = round(span_y)
    tolerance = 1.0e-9 * max(1.0, abs(span_x), abs(span_y))
    if (abs(span_x - rounded_x) > tolerance or
            abs(span_y - rounded_y) > tolerance):
        raise ValueError(
            'gas field bounds must be separated by whole resolution steps')
    width = int(rounded_x) + 1
    height = int(rounded_y) + 1
    geometry = GridGeometry(
        width, height, resolution,
        min_x - 0.5 * resolution, min_y - 0.5 * resolution)
    geometry.validate()
    return geometry


def sampling_mask(map_message, clearance, start_x, start_y):
    """Return free cells satisfying clearance and start connectivity."""
    geometry = GridGeometry.from_message(map_message)
    clearance = float(clearance)
    if clearance < 0.0:
        raise ValueError('sampling clearance must be non-negative')
    occupancy = np.asarray(map_message.data, dtype=np.int16)
    if occupancy.size != geometry.width * geometry.height:
        raise ValueError('occupancy data size does not match map geometry')
    free = occupancy.reshape(geometry.height, geometry.width) == 0
    if clearance > 0.0:
        distances = distance_transform_edt(free) * geometry.resolution
        free &= distances + 1.0e-12 >= clearance
    if not np.any(free):
        raise ValueError('sampling domain has no cells after clearance')

    start_row, start_col = geometry.world_to_cell(start_x, start_y)
    if not geometry.contains_cell(start_row, start_col) or not free[
            start_row, start_col]:
        raise ValueError('robot start is outside the sampling domain')
    # Four-connected reachability avoids treating two cells that only touch at
    # an obstacle corner as a traversable robot path.
    connectivity = np.asarray([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.int8)
    components, _ = label(free, structure=connectivity)
    component = int(components[start_row, start_col])
    if component == 0:
        raise ValueError('robot start has no connected sampling component')
    return components == component


def navigation_goal_mask(map_message, traversal_mask, inflation_radius):
    """Restrict traversal cells to positions outside Nav2 inflation."""
    geometry = GridGeometry.from_message(map_message)
    radius = float(inflation_radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            'navigation goal inflation radius must be finite and positive')
    traversal = np.asarray(traversal_mask, dtype=bool)
    expected_shape = (geometry.height, geometry.width)
    if traversal.shape != expected_shape:
        raise ValueError(
            f'traversal mask shape {traversal.shape} does not match '
            f'geometry {expected_shape}')
    occupancy = np.asarray(map_message.data, dtype=np.int16)
    if occupancy.size != geometry.width * geometry.height:
        raise ValueError('occupancy data size does not match map geometry')
    free = occupancy.reshape(geometry.height, geometry.width) == 0
    distances = distance_transform_edt(free) * geometry.resolution
    goals = traversal & (distances > radius)
    if not np.any(goals):
        raise ValueError(
            'navigation goal domain has no cells outside inflation')
    return goals


def mask_contains_world(mask, geometry, x, y):
    row, col = geometry.world_to_cell(x, y)
    return bool(geometry.contains_cell(row, col) and mask[row, col])


__all__ = [
    'field_geometry', 'mask_contains_world', 'navigation_goal_mask',
    'sampling_mask']
