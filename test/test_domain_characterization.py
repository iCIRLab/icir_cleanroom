"""Characterize the separate gas-field and sampling-domain contracts."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from icir_cleanroom.gas_mapping.application.hrs import HrsManager
from icir_cleanroom.gas_mapping.environment import (
    random_source_is_lrs_detectable)
from icir_cleanroom.gas_mapping.mapping.domains import (
    field_geometry, navigation_goal_mask, sampling_mask)
from icir_cleanroom.gas_mapping.mapping.gmrf import GmrfGrid
from icir_cleanroom.gas_mapping.mapping.grid_geometry import GridGeometry
from icir_cleanroom.gas_mapping.mapping.kmeans_partition import (
    partition_sampling_cells)
from icir_cleanroom.gas_mapping.mapping.lrs_representatives import (
    build_lrs_representatives)
from icir_cleanroom.gas_mapping.mapping.path_distance import (
    SamplingDistanceOracle)
from icir_cleanroom.gas_mapping.ros.lrs_planner_node import (
    LrsPathPlannerNode)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def load_environment_map(profile_name, map_message_factory):
    profile = yaml.safe_load((
        PACKAGE_ROOT / 'config' / 'environments' /
        f'{profile_name}.yaml').read_text(encoding='utf-8'))
    environment = profile['gas_environment']
    map_yaml_path = PACKAGE_ROOT / profile['environment']['map']
    map_yaml = yaml.safe_load(map_yaml_path.read_text(encoding='utf-8'))
    image = np.asarray(Image.open(
        map_yaml_path.parent / map_yaml['image']).convert('L'))
    occupancy_probability = (
        image.astype(float) / 255.0 if map_yaml['negate']
        else (255.0 - image.astype(float)) / 255.0)
    occupancy = np.flipud(np.where(
        occupancy_probability > map_yaml['occupied_thresh'], 100,
        np.where(
            occupancy_probability < map_yaml['free_thresh'], 0, -1)))
    navigation_map = map_message_factory(
        data=occupancy.ravel(), width=occupancy.shape[1],
        height=occupancy.shape[0], resolution=map_yaml['resolution'],
        origin_x=map_yaml['origin'][0], origin_y=map_yaml['origin'][1])
    return environment, navigation_map


def test_grid_geometry_round_trip_supports_rotated_origin():
    geometry = GridGeometry(4, 3, 0.5, -2.0, 1.0, math.pi / 2.0)
    x, y = geometry.cell_center(1, 2)
    assert geometry.world_to_cell(x, y) == (1, 2)


def test_field_bounds_are_first_and_last_cell_centers():
    empty = field_geometry(0.0, 50.0, 0.0, 50.0, 1.0)
    warehouse = field_geometry(-7.0, 7.0, -10.0, 10.0, 0.5)

    assert (empty.width, empty.height) == (51, 51)
    assert empty.cell_center(0, 0) == (0.0, 0.0)
    assert empty.cell_center(50, 50) == (50.0, 50.0)
    assert (warehouse.width, warehouse.height) == (29, 41)
    assert warehouse.cell_center(0, 0) == (-7.0, -10.0)
    assert warehouse.cell_center(40, 28) == (7.0, 10.0)


def test_field_bounds_require_whole_resolution_steps():
    with pytest.raises(ValueError, match='whole resolution steps'):
        field_geometry(0.0, 1.1, 0.0, 1.0, 0.5)


def test_sampling_distance_does_not_cut_diagonal_obstacle_corners():
    geometry = GridGeometry(3, 3, 1.0, 0.0, 0.0)
    free = np.asarray([
        [True, True, True],
        [True, False, True],
        [True, True, True],
    ])
    oracle = SamplingDistanceOracle(geometry, free)

    distance = oracle.distance((0.5, 0.5), (2.5, 2.5))

    assert math.isclose(distance, 4.0)


def test_sampling_distance_rejects_disconnected_points():
    geometry = GridGeometry(3, 1, 1.0, 0.0, 0.0)
    oracle = SamplingDistanceOracle(
        geometry, np.asarray([[True, False, True]]))

    with pytest.raises(ValueError, match='no sampling-domain path'):
        oracle.distance((0.5, 0.5), (2.5, 0.5))


def test_obstacle_is_in_field_but_not_sampling_domain(map_message_factory):
    navigation_map = map_message_factory(
        data=[0, 0, 0, 0, 100, 0, 0, 0, 0], width=3, height=3)
    mask = sampling_mask(
        navigation_map, clearance=0.0, start_x=0.5, start_y=0.5)
    assert not mask[1, 1]

    field_map = map_message_factory(width=3, height=3)
    gmrf = GmrfGrid(field_map)
    assert len(gmrf.var_cells) == 9
    assert gmrf.variable_at(1, 1) is not None


def test_sampling_domain_keeps_only_start_component(map_message_factory):
    navigation_map = map_message_factory(
        data=[0, 100, 0, 0, 100, 0, 0, 100, 0], width=3, height=3)
    mask = sampling_mask(
        navigation_map, clearance=0.0, start_x=0.5, start_y=0.5)
    assert np.array_equal(mask[:, 0], [True, True, True])
    assert not np.any(mask[:, 2])


def test_sampling_domain_does_not_cross_diagonal_obstacle_corner(
        map_message_factory):
    navigation_map = map_message_factory(
        data=[0, 100, 100, 0], width=2, height=2)
    mask = sampling_mask(
        navigation_map, clearance=0.0, start_x=0.5, start_y=0.5)
    assert mask[0, 0]
    assert not mask[1, 1]


def test_navigation_goal_domain_is_strict_sampling_subset(
        map_message_factory):
    navigation_map = map_message_factory(
        data=[100, 0, 0, 0, 0], width=5, height=1)
    traversal = sampling_mask(
        navigation_map, clearance=0.0, start_x=1.5, start_y=0.5)

    goals = navigation_goal_mask(
        navigation_map, traversal, inflation_radius=2.0)

    assert np.all(~goals | traversal)
    assert np.array_equal(goals, [[False, False, False, True, True]])


def test_narrow_passage_remains_traversable_but_is_not_a_goal(
        map_message_factory):
    occupancy = np.full((7, 9), 100, dtype=np.int8)
    occupancy[1:6, 1:8] = 0
    occupancy[1:3, 4] = 100
    occupancy[4:6, 4] = 100
    navigation_map = map_message_factory(
        data=occupancy.ravel(), width=9, height=7)
    traversal = sampling_mask(
        navigation_map, clearance=0.0, start_x=2.5, start_y=3.5)

    goals = navigation_goal_mask(
        navigation_map, traversal, inflation_radius=1.1)

    assert traversal[3, 4]
    assert not goals[3, 4]
    assert goals[3, 2]
    assert goals[3, 6]


def test_lrs_uses_goal_domain_for_stops_and_sampling_domain_for_distance(
        map_message_factory):
    sampling = map_message_factory(width=5, height=1)
    goals = map_message_factory(
        data=[0, 100, 0, 100, 0], width=5, height=1)
    gmrf = map_message_factory(width=5, height=1)
    planner = object.__new__(LrsPathPlannerNode)
    planner.lrs_cluster_count = 3
    planner.lrs_cluster_random_seed = 0

    selection, points, oracle = planner.build_points(
        sampling, goals, gmrf)

    assert len(selection.partition.centroids) == 3
    assert not selection.omitted_cluster_ids
    assert [(point.x, point.y) for point in points] == [
        (0.5, 0.5), (2.5, 0.5), (4.5, 0.5)]
    assert math.isclose(oracle.distance((0.5, 0.5), (4.5, 0.5)), 4.0)


def test_lrs_rejects_goal_cells_outside_sampling_domain(
        map_message_factory):
    sampling = map_message_factory(
        data=[0, 100, 0], width=3, height=1)
    goals = map_message_factory(width=3, height=1)
    gmrf = map_message_factory(width=3, height=1)
    planner = object.__new__(LrsPathPlannerNode)
    planner.lrs_cluster_count = 3
    planner.lrs_cluster_random_seed = 0

    with pytest.raises(ValueError, match='must be a sampling domain subset'):
        planner.build_points(sampling, goals, gmrf)


def test_aws_navigation_goal_domain_keeps_ten_safe_representatives(
        map_message_factory):
    profile = yaml.safe_load((
        PACKAGE_ROOT / 'config' / 'environments' /
        'aws_small_warehouse.yaml').read_text(encoding='utf-8'))
    environment, navigation_map = load_environment_map(
        'aws_small_warehouse', map_message_factory)
    traversal = sampling_mask(
        navigation_map, environment['sampling_clearance'],
        environment['robot_start_x'], environment['robot_start_y'])
    goals = navigation_goal_mask(
        navigation_map, traversal, inflation_radius=0.45)
    gmrf = field_geometry(
        environment['map_min_x'], environment['map_max_x'],
        environment['map_min_y'], environment['map_max_y'],
        environment['gmrf_resolution'])
    navigation_geometry = GridGeometry.from_message(navigation_map)
    lrs = profile['lrs']

    selection = build_lrs_representatives(
        gmrf, navigation_geometry, traversal, goals,
        lrs['lrs_cluster_count'], lrs['lrs_cluster_random_seed'])

    assert len(selection.representatives) == 10
    assert not selection.omitted_cluster_ids
    for representative in selection.representatives:
        row, col = navigation_geometry.world_to_cell(
            representative.x, representative.y)
        assert traversal[row, col]
        assert goals[row, col]


def test_aws_sampling_domain_uses_configured_cluster_count(
        map_message_factory):
    profile = yaml.safe_load((
        PACKAGE_ROOT / 'config' / 'environments' /
        'aws_small_warehouse.yaml').read_text(encoding='utf-8'))
    environment, navigation_map = load_environment_map(
        'aws_small_warehouse', map_message_factory)
    traversal = sampling_mask(
        navigation_map, environment['sampling_clearance'],
        environment['robot_start_x'], environment['robot_start_y'])
    gmrf = field_geometry(
        environment['map_min_x'], environment['map_max_x'],
        environment['map_min_y'], environment['map_max_y'],
        environment['gmrf_resolution'])
    navigation_geometry = GridGeometry.from_message(navigation_map)

    result = partition_sampling_cells(
        gmrf, navigation_geometry, traversal,
        cluster_count=profile['lrs']['lrs_cluster_count'],
        random_seed=profile['lrs']['lrs_cluster_random_seed'])

    assert (gmrf.width, gmrf.height, gmrf.resolution) == (29, 41, 0.5)
    assert len(result.centroids) == profile['lrs']['lrs_cluster_count']
    assert sorted({cell.cluster_id for cell in result.cells}) == list(
        range(profile['lrs']['lrs_cluster_count']))
    for cell in result.cells:
        row, col = navigation_geometry.world_to_cell(cell.x, cell.y)
        assert traversal[row, col]


def test_empty_navigation_goal_domain_preserves_36_lrs_points(
        map_message_factory):
    profile = yaml.safe_load((
        PACKAGE_ROOT / 'config' / 'environments' /
        'empty_50m.yaml').read_text(encoding='utf-8'))
    environment, navigation_map = load_environment_map(
        'empty_50m', map_message_factory)
    traversal = sampling_mask(
        navigation_map, environment['sampling_clearance'],
        environment['robot_start_x'], environment['robot_start_y'])
    goals = navigation_goal_mask(
        navigation_map, traversal, inflation_radius=0.25)
    gmrf = field_geometry(
        environment['map_min_x'], environment['map_max_x'],
        environment['map_min_y'], environment['map_max_y'],
        environment['gmrf_resolution'])

    lrs = profile['lrs']
    selection = build_lrs_representatives(
        gmrf, GridGeometry.from_message(navigation_map), traversal, goals,
        lrs['lrs_cluster_count'], lrs['lrs_cluster_random_seed'])

    assert len(selection.representatives) == 36
    assert not selection.omitted_cluster_ids


def test_hrs_candidates_are_restricted_without_removing_field_variables(
        map_message_factory):
    gmrf = GmrfGrid(map_message_factory(width=3, height=1))
    gmrf.solution[:] = [0.1, 0.9, 0.2]
    manager = HrsManager()
    candidates = manager.build_candidates(
        gmrf, sampled_variables=set(), ucb_coefficient=0.0, count=3,
        eligible_variables={0, 2})
    assert [candidate.variable for candidate in candidates] == [2, 0]
    assert len(gmrf.var_cells) == 3


def test_hrs_candidates_prefer_nearer_cells_when_ucb_is_tied(
        map_message_factory):
    gmrf = GmrfGrid(map_message_factory(width=3, height=1))
    gmrf.solution[:] = [0.5, 0.5, 0.5]
    gmrf.variance[:] = [0.0, 0.0, 0.0]
    manager = HrsManager()
    candidates = manager.build_candidates(
        gmrf, sampled_variables=set(), ucb_coefficient=0.0, count=3,
        current_xy=(0.5, 0.5), distance_weight=0.5)
    assert [candidate.variable for candidate in candidates] == [0, 1, 2]
    assert candidates[0].x == 0.5
    assert candidates[0].y == 0.5


def test_source_detectability_uses_accessible_sampling_points_only():
    source = {
        'source_x': 5.0, 'source_y': 5.0, 'source_sigma': 1.0}
    assert not random_source_is_lrs_detectable(
        source, 0.2, sampling_points=[(0.0, 0.0)])
    assert random_source_is_lrs_detectable(
        source, 0.2, sampling_points=[(4.0, 5.0)])
