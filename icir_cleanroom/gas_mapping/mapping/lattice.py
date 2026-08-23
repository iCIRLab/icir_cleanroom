"""Shared construction of coarse sampling points from field-cell indices."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LatticePoint:
    lattice_row: int
    lattice_col: int
    field_row: int
    field_col: int
    x: float
    y: float


def inclusive_axis(minimum, maximum, spacing):
    minimum = float(minimum)
    maximum = float(maximum)
    spacing = float(spacing)
    if not all(math.isfinite(value) for value in (
            minimum, maximum, spacing)):
        raise ValueError('lattice bounds and spacing must be finite')
    if maximum < minimum:
        raise ValueError('lattice maximum must not be less than minimum')
    if spacing <= 0.0:
        raise ValueError('lattice spacing must be positive')
    count = int(math.floor((maximum - minimum) / spacing + 1.0e-9))
    return [minimum + index * spacing for index in range(count + 1)]


def _center_cell(geometry, x, y, name):
    row, col = geometry.world_to_cell(x, y)
    if not geometry.contains_cell(row, col):
        raise ValueError(f'{name} is outside the GMRF field')
    center_x, center_y = geometry.cell_center(row, col)
    tolerance = 1.0e-9 * max(
        1.0, geometry.resolution, abs(float(x)), abs(float(y)))
    if (abs(center_x - float(x)) > tolerance or
            abs(center_y - float(y)) > tolerance):
        raise ValueError(f'{name} must be a GMRF cell center')
    return row, col


def field_cell_lattice(
        field_geometry, min_x, max_x, min_y, max_y, stride_cells):
    """Create a coarse lattice directly from GMRF cell indices."""
    if isinstance(stride_cells, bool) or not isinstance(stride_cells, int):
        raise ValueError('LRS stride must be an integer number of GMRF cells')
    if stride_cells <= 0:
        raise ValueError('LRS stride must be positive')
    min_row, min_col = _center_cell(
        field_geometry, min_x, min_y, 'LRS minimum')
    max_row, max_col = _center_cell(
        field_geometry, max_x, max_y, 'LRS maximum')
    if max_row < min_row or max_col < min_col:
        raise ValueError('LRS maximums must not be below minimums')

    points = []
    for lattice_row, field_row in enumerate(
            range(min_row, max_row + 1, stride_cells)):
        for lattice_col, field_col in enumerate(
                range(min_col, max_col + 1, stride_cells)):
            x, y = field_geometry.cell_center(field_row, field_col)
            points.append(LatticePoint(
                lattice_row=lattice_row, lattice_col=lattice_col,
                field_row=field_row, field_col=field_col, x=x, y=y))
    return points


def sampling_lattice(
        field_geometry, sampling_geometry, sampling_mask,
        min_x, max_x, min_y, max_y, stride_cells):
    """Filter a GMRF-index lattice to robot-accessible sampling cells."""
    mask = np.asarray(sampling_mask, dtype=bool)
    expected_shape = (sampling_geometry.height, sampling_geometry.width)
    if mask.shape != expected_shape:
        raise ValueError(
            f'sampling mask shape {mask.shape} does not match '
            f'geometry {expected_shape}')

    points = field_cell_lattice(
        field_geometry, min_x, max_x, min_y, max_y, stride_cells)
    return [point for point in points if _is_accessible(
        point, sampling_geometry, mask)]


def _is_accessible(point, sampling_geometry, sampling_mask):
    row, col = sampling_geometry.world_to_cell(point.x, point.y)
    return bool(
        sampling_geometry.contains_cell(row, col) and
        sampling_mask[row, col])


__all__ = [
    'LatticePoint', 'field_cell_lattice', 'inclusive_axis',
    'sampling_lattice']
