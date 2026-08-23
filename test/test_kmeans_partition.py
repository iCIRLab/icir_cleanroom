"""Characterize Euclidean K-means partitioning of the sampling ROI."""

import math

import numpy as np
import pytest

from icir_cleanroom.gas_mapping.mapping.grid_geometry import GridGeometry
from icir_cleanroom.gas_mapping.mapping.kmeans_partition import (
    partition_sampling_cells)


def partition_fixture(seed=7):
    field = GridGeometry(8, 8, 1.0, 0.0, 0.0)
    sampling = GridGeometry(8, 8, 1.0, 0.0, 0.0)
    mask = np.ones((8, 8), dtype=bool)
    mask[3:5, 3:5] = False
    return partition_sampling_cells(
        field, sampling, mask, cluster_count=4, random_seed=seed)


def test_partition_assigns_each_accessible_cell_to_one_non_empty_cluster():
    result = partition_fixture()

    assert len(result.cells) == 60
    assert sorted({cell.cluster_id for cell in result.cells}) == [0, 1, 2, 3]
    assert not any(
        3 <= cell.field_row < 5 and 3 <= cell.field_col < 5
        for cell in result.cells)


def test_partition_centroids_are_member_coordinate_means():
    result = partition_fixture()

    for cluster_id, centroid in enumerate(result.centroids):
        coordinates = np.asarray([
            (cell.x, cell.y) for cell in result.cells
            if cell.cluster_id == cluster_id])
        assert np.allclose(centroid, coordinates.mean(axis=0))
    assert math.isfinite(result.inertia)
    assert result.inertia >= 0.0


def test_partition_is_deterministic_and_cluster_ids_are_canonical():
    first = partition_fixture(seed=11)
    second = partition_fixture(seed=11)

    assert first == second
    assert list(first.centroids) == sorted(
        first.centroids, key=lambda point: (point[0], point[1]))


@pytest.mark.parametrize('cluster_count', [0, -1, 1.5, True])
def test_partition_rejects_invalid_cluster_count(cluster_count):
    geometry = GridGeometry(2, 2, 1.0, 0.0, 0.0)

    with pytest.raises(ValueError, match='cluster_count'):
        partition_sampling_cells(
            geometry, geometry, np.ones((2, 2), dtype=bool),
            cluster_count=cluster_count)


def test_partition_rejects_more_clusters_than_accessible_cells():
    geometry = GridGeometry(2, 2, 1.0, 0.0, 0.0)
    mask = np.asarray([[True, False], [False, True]])

    with pytest.raises(ValueError, match='exceeds accessible'):
        partition_sampling_cells(
            geometry, geometry, mask, cluster_count=3)


def test_partition_rejects_invalid_sampling_mask_shape():
    geometry = GridGeometry(2, 2, 1.0, 0.0, 0.0)

    with pytest.raises(ValueError, match='sampling mask shape'):
        partition_sampling_cells(
            geometry, geometry, np.ones((1, 2), dtype=bool),
            cluster_count=1)
