"""K-means partitioning of accessible gas-field cells."""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
from scipy.cluster.vq import ClusterError, kmeans2


@dataclass(frozen=True)
class ClusterCell:
    field_row: int
    field_col: int
    x: float
    y: float
    cluster_id: int


@dataclass(frozen=True)
class KMeansPartition:
    cells: tuple
    centroids: tuple
    inertia: float


def _candidate_cells(field_geometry, sampling_geometry, sampling_mask):
    field_geometry.validate()
    sampling_geometry.validate()
    mask = np.asarray(sampling_mask, dtype=bool)
    expected_shape = (sampling_geometry.height, sampling_geometry.width)
    if mask.shape != expected_shape:
        raise ValueError(
            f'sampling mask shape {mask.shape} does not match '
            f'geometry {expected_shape}')

    cells = []
    for field_row in range(field_geometry.height):
        for field_col in range(field_geometry.width):
            x, y = field_geometry.cell_center(field_row, field_col)
            sampling_row, sampling_col = sampling_geometry.world_to_cell(x, y)
            if (sampling_geometry.contains_cell(sampling_row, sampling_col) and
                    mask[sampling_row, sampling_col]):
                cells.append((field_row, field_col, float(x), float(y)))
    return cells


def _canonicalize(centroids, labels):
    order = np.lexsort((centroids[:, 1], centroids[:, 0]))
    inverse = np.empty(len(order), dtype=np.int32)
    inverse[order] = np.arange(len(order), dtype=np.int32)
    return centroids[order], inverse[np.asarray(labels, dtype=np.int32)]


def partition_sampling_cells(
        field_geometry, sampling_geometry, sampling_mask,
        cluster_count, random_seed=0, restarts=10,
        max_iterations=100, tolerance=1.0e-6):
    """Partition accessible GMRF cell centers by Euclidean K-means."""
    if (isinstance(cluster_count, bool) or
            not isinstance(cluster_count, Integral) or cluster_count <= 0):
        raise ValueError('cluster_count must be a positive integer')
    if (isinstance(random_seed, bool) or
            not isinstance(random_seed, Integral)):
        raise ValueError('random_seed must be an integer')
    if (isinstance(restarts, bool) or not isinstance(restarts, Integral) or
            restarts <= 0):
        raise ValueError('restarts must be a positive integer')
    if (isinstance(max_iterations, bool) or
            not isinstance(max_iterations, Integral) or
            max_iterations <= 0):
        raise ValueError('max_iterations must be a positive integer')
    if not math.isfinite(float(tolerance)) or tolerance <= 0.0:
        raise ValueError('tolerance must be positive and finite')

    candidates = _candidate_cells(
        field_geometry, sampling_geometry, sampling_mask)
    if cluster_count > len(candidates):
        raise ValueError(
            f'cluster_count {cluster_count} exceeds accessible GMRF '
            f'cell count {len(candidates)}')
    coordinates = np.asarray([
        (cell[2], cell[3]) for cell in candidates], dtype=float)
    if coordinates.size == 0 or not np.all(np.isfinite(coordinates)):
        raise ValueError('accessible GMRF cell coordinates must be finite')

    best = None
    for attempt in range(int(restarts)):
        try:
            centroids, labels = kmeans2(
                coordinates, int(cluster_count), iter=int(max_iterations),
                thresh=float(tolerance), minit='++', missing='raise',
                seed=int(random_seed) + attempt)
        except ClusterError:
            continue
        if len(np.unique(labels)) != cluster_count:
            continue
        centroids, labels = _canonicalize(centroids, labels)
        residuals = coordinates - centroids[labels]
        inertia = float(np.sum(residuals * residuals))
        signature = tuple(centroids.ravel())
        candidate = (inertia, signature, centroids, labels)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if best is None:
        raise RuntimeError(
            'K-means could not produce the requested non-empty clusters')
    inertia, _, centroids, labels = best
    cells = tuple(ClusterCell(
        field_row=cell[0], field_col=cell[1], x=cell[2], y=cell[3],
        cluster_id=int(labels[index]))
        for index, cell in enumerate(candidates))
    return KMeansPartition(
        cells=cells,
        centroids=tuple(
            (float(centroid[0]), float(centroid[1]))
            for centroid in centroids),
        inertia=inertia)


__all__ = [
    'ClusterCell', 'KMeansPartition', 'partition_sampling_cells']
