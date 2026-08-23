"""Select safe representative LRS cells from K-means partitions."""

from dataclasses import dataclass

import numpy as np

from .kmeans_partition import (
    ClusterCell, KMeansPartition, partition_sampling_cells)


@dataclass(frozen=True)
class LrsRepresentativeSelection:
    partition: KMeansPartition
    representatives: tuple[ClusterCell, ...]
    omitted_cluster_ids: tuple[int, ...]


def _validated_mask(mask, geometry, description):
    values = np.asarray(mask, dtype=bool)
    expected_shape = (geometry.height, geometry.width)
    if values.shape != expected_shape:
        raise ValueError(
            f'{description} shape {values.shape} does not match '
            f'geometry {expected_shape}')
    return values


def select_cluster_representatives(
        partition, domain_geometry, navigation_goal_mask):
    """Select the centroid-nearest safe member of every cluster."""
    goals = _validated_mask(
        navigation_goal_mask, domain_geometry, 'navigation goal mask')
    representatives = []
    omitted = []
    for cluster_id, centroid in enumerate(partition.centroids):
        eligible = []
        for cell in partition.cells:
            if cell.cluster_id != cluster_id:
                continue
            row, col = domain_geometry.world_to_cell(cell.x, cell.y)
            if (domain_geometry.contains_cell(row, col) and goals[row, col]):
                distance_sq = (
                    (cell.x - centroid[0]) ** 2 +
                    (cell.y - centroid[1]) ** 2)
                eligible.append((
                    distance_sq, cell.field_row, cell.field_col, cell))
        if not eligible:
            omitted.append(cluster_id)
            continue
        representatives.append(min(eligible)[-1])
    return LrsRepresentativeSelection(
        partition=partition,
        representatives=tuple(representatives),
        omitted_cluster_ids=tuple(omitted))


def build_lrs_representatives(
        field_geometry, domain_geometry, sampling_mask,
        navigation_goal_mask, cluster_count, random_seed=0):
    """Build one safe LRS representative per eligible K-means cluster."""
    sampling = _validated_mask(
        sampling_mask, domain_geometry, 'sampling mask')
    goals = _validated_mask(
        navigation_goal_mask, domain_geometry, 'navigation goal mask')
    if np.any(goals & ~sampling):
        raise ValueError(
            'navigation goal mask must be a sampling mask subset')
    partition = partition_sampling_cells(
        field_geometry, domain_geometry, sampling,
        cluster_count, random_seed)
    return select_cluster_representatives(
        partition, domain_geometry, goals)


__all__ = [
    'LrsRepresentativeSelection', 'build_lrs_representatives',
    'select_cluster_representatives']
