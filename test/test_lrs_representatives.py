"""Characterize safe representative selection for K-means LRS cells."""

import numpy as np
import pytest

from icir_cleanroom.gas_mapping.mapping.grid_geometry import GridGeometry
from icir_cleanroom.gas_mapping.mapping.kmeans_partition import (
    ClusterCell, KMeansPartition)
from icir_cleanroom.gas_mapping.mapping.lrs_representatives import (
    build_lrs_representatives, select_cluster_representatives)


def test_representative_is_nearest_navigation_goal_cluster_member():
    geometry = GridGeometry(4, 2, 1.0, 0.0, 0.0)
    partition = KMeansPartition(
        cells=(
            ClusterCell(0, 0, 0.5, 0.5, 0),
            ClusterCell(0, 1, 1.5, 0.5, 0),
            ClusterCell(0, 2, 2.5, 0.5, 0),
            ClusterCell(1, 3, 3.5, 1.5, 1)),
        centroids=((1.7, 0.5), (3.5, 1.5)), inertia=1.0)
    goals = np.zeros((2, 4), dtype=bool)
    goals[0, 0] = goals[0, 2] = goals[1, 3] = True

    result = select_cluster_representatives(partition, geometry, goals)

    assert [(cell.cluster_id, cell.field_row, cell.field_col)
            for cell in result.representatives] == [(0, 0, 2), (1, 1, 3)]
    assert result.omitted_cluster_ids == ()


def test_representative_tie_breaks_by_field_cell_and_omits_unsafe_cluster():
    geometry = GridGeometry(3, 2, 1.0, 0.0, 0.0)
    partition = KMeansPartition(
        cells=(
            ClusterCell(0, 0, 0.5, 0.5, 0),
            ClusterCell(0, 1, 1.5, 0.5, 0),
            ClusterCell(1, 2, 2.5, 1.5, 1)),
        centroids=((1.0, 0.5), (2.5, 1.5)), inertia=0.5)
    goals = np.zeros((2, 3), dtype=bool)
    goals[0, 0] = goals[0, 1] = True

    result = select_cluster_representatives(partition, geometry, goals)

    assert [(cell.field_row, cell.field_col)
            for cell in result.representatives] == [(0, 0)]
    assert result.omitted_cluster_ids == (1,)


def test_builder_uses_sampling_cells_and_navigation_goal_subset():
    field = GridGeometry(6, 2, 1.0, 0.0, 0.0)
    domain = GridGeometry(6, 2, 1.0, 0.0, 0.0)
    sampling = np.ones((2, 6), dtype=bool)
    sampling[0, 0] = False
    goals = sampling.copy()
    goals[:, 2] = False

    first = build_lrs_representatives(
        field, domain, sampling, goals, cluster_count=3, random_seed=5)
    second = build_lrs_representatives(
        field, domain, sampling, goals, cluster_count=3, random_seed=5)

    assert first == second
    assert len(first.representatives) == 3
    for representative in first.representatives:
        row, col = domain.world_to_cell(
            representative.x, representative.y)
        assert sampling[row, col]
        assert goals[row, col]
        assert representative in first.partition.cells


def test_builder_rejects_goal_cells_outside_sampling_domain():
    geometry = GridGeometry(2, 1, 1.0, 0.0, 0.0)
    sampling = np.asarray([[True, False]])
    goals = np.asarray([[True, True]])

    with pytest.raises(ValueError, match='must be a sampling mask subset'):
        build_lrs_representatives(
            geometry, geometry, sampling, goals,
            cluster_count=1, random_seed=0)
