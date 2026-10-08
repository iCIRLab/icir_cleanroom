"""The cleanroom_amc environment, its AMC robot, and the shared robot selection."""
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml
from PIL import Image

from icir_cleanroom.gas_mapping.mapping.domains import (
    field_geometry, navigation_goal_mask, sampling_mask)
from icir_cleanroom.gas_mapping.mapping.grid_geometry import GridGeometry
from icir_cleanroom.gas_mapping.navigation_profile import (
    DEFAULT_ROBOT, load_navigation_settings, navigation_goal_clearance,
    render_robot_sdf, validated_robot)

PACKAGE = Path(__file__).resolve().parents[1]
ENVIRONMENT = 'cleanroom_amc'


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


@pytest.fixture(scope='module')
def profile():
    return load_yaml(PACKAGE/'config'/'environments'/f'{ENVIRONMENT}.yaml')


@pytest.fixture(scope='module')
def occupancy():
    """Rebuild the map the way the launch and nodes consume it."""
    map_yaml = PACKAGE/'maps'/'cleanroom_amc'/'cleanroom.yaml'
    meta = load_yaml(map_yaml)
    pixels = np.asarray(
        Image.open(map_yaml.parent/meta['image']).convert('L')).astype(float)
    probability = (pixels if meta['negate'] else 255. - pixels)/255.
    grid = np.flipud(np.where(
        probability > meta['occupied_thresh'], 100,
        np.where(probability < meta['free_thresh'], 0, -1)))
    message = NS(data=grid.ravel(), info=NS(
        width=grid.shape[1], height=grid.shape[0],
        resolution=float(np.float32(meta['resolution'])),
        origin=NS(position=NS(x=meta['origin'][0], y=meta['origin'][1]))))
    return message, meta


def test_profile_paths_exist_and_are_package_relative(profile):
    environment = profile['environment']
    for key in ('world', 'map', 'navigation_profile'):
        value = environment[key]
        assert not os.path.isabs(value), f'{key} must be package relative'
        assert 'Documents' not in value and 'Codex' not in value
        assert (PACKAGE/value).is_file(), f'{key} missing: {value}'


def test_robot_block_selects_the_amc_model(profile):
    robot = validated_robot(profile['environment'].get('robot'))
    assert robot['entity'] == 'mobile_amc'
    assert robot['lidar_sensor_name'] == 'amc_lidar'
    for key in ('description', 'model'):
        assert (PACKAGE/robot[key]).is_file()


def test_existing_environments_keep_the_turtlebot3_default():
    for name in ('empty_50m', 'aws_small_warehouse'):
        environment = load_yaml(
            PACKAGE/'config'/'environments'/f'{name}.yaml')['environment']
        assert validated_robot(environment.get('robot')) == DEFAULT_ROBOT


def test_spawn_matches_gas_and_lrs_start(profile):
    spawn = profile['environment']['robot_spawn']
    assert (spawn['x'], spawn['y']) == (-8.0, -5.0)
    for section in ('gas_environment', 'lrs'):
        assert profile[section]['robot_start_x'] == spawn['x']
        assert profile[section]['robot_start_y'] == spawn['y']


def test_gas_field_is_the_room_not_the_map_extent(profile, occupancy):
    _, meta = occupancy
    gas = profile['gas_environment']
    assert (gas['map_min_x'], gas['map_max_x']) == (-10.0, 10.0)
    assert (gas['map_min_y'], gas['map_max_y']) == (-7.0, 7.0)
    assert meta['resolution'] == 0.05
    assert meta['origin'] == [-10.5, -7.5, 0.0]
    message, _ = occupancy
    extent_x = meta['origin'][0] + message.info.width*message.info.resolution
    extent_y = meta['origin'][1] + message.info.height*message.info.resolution
    # The gas field must sit inside the map, and must not be the map itself.
    assert meta['origin'][0] < gas['map_min_x'] and gas['map_max_x'] < extent_x
    assert meta['origin'][1] < gas['map_min_y'] and gas['map_max_y'] < extent_y


def test_sampling_domain_reaches_the_main_room_through_the_airlock(
        profile, occupancy):
    message, _ = occupancy
    gas = profile['gas_environment']
    geometry = GridGeometry.from_message(message)
    mask = sampling_mask(
        message, gas['sampling_clearance'],
        gas['robot_start_x'], gas['robot_start_y'])
    # The spawn is inside the airlock; (5, 0) is the far side of the main room.
    assert mask[geometry.world_to_cell(5.0, 0.0)]
    # The only route is the 1.6 m doorway at x = -6.5.
    column = geometry.world_to_cell(-6.5, 0.0)[1]
    low = geometry.world_to_cell(0.0, -6.0)[0]
    high = geometry.world_to_cell(0.0, -4.0)[0]
    band = sum(bool(mask[row, column])
               for row in range(min(low, high), max(low, high) + 1))
    assert band > 0, 'airlock doorway is closed to the sampling domain'
    assert band*message.info.resolution >= 0.5, 'doorway band is too narrow'


def test_navigation_goal_domain_is_not_empty_for_the_larger_robot(
        profile, occupancy):
    message, _ = occupancy
    gas = profile['gas_environment']
    nav2_config, _ = load_navigation_settings(
        PACKAGE/'config'/'nav2_params.yaml',
        PACKAGE/profile['environment']['navigation_profile'])
    clearance = navigation_goal_clearance(nav2_config)
    assert clearance == pytest.approx(0.5)
    mask = sampling_mask(
        message, gas['sampling_clearance'],
        gas['robot_start_x'], gas['robot_start_y'])
    goals = navigation_goal_mask(message, mask, clearance)
    assert goals.sum() > 0
    # The LRS needs enough goal cells to place one representative per cluster.
    assert goals.sum() >= profile['lrs']['lrs_cluster_count']


def test_navigation_profile_uses_a_footprint_sized_for_the_amc(profile):
    nav2_config, motion = load_navigation_settings(
        PACKAGE/'config'/'nav2_params.yaml',
        PACKAGE/profile['environment']['navigation_profile'])
    for name in ('local_costmap', 'global_costmap'):
        parameters = nav2_config[name][name]['ros__parameters']
        corners = yaml.safe_load(parameters['footprint'])
        assert len(corners) == 4
        length = max(x for x, _ in corners) - min(x for x, _ in corners)
        width = max(y for _, y in corners) - min(y for _, y in corners)
        assert length == pytest.approx(0.96) and width == pytest.approx(0.68)
        # Inflation covers the padded inscribed radius; the rectangular
        # footprint still governs collision checks at the robot's corners.
        radius = min(length, width)/2 + parameters['footprint_padding']
        assert parameters['inflation_layer']['inflation_radius'] > radius
    assert motion == {'max_wheel_acceleration': 0.6, 'lidar_update_rate': 10.0}


def test_amc_sdf_keeps_the_gas_sensor_link_that_the_plugin_resolves():
    sdf = (PACKAGE/'urdf'/'mobile_amc.sdf').read_text()
    links = re.findall(r"<link name='([^']*)'", sdf)
    assert 'gas_sensor_link' in links, (
        'fixed-joint lumping removed the link the gas sensor plugin needs')
    plugins = dict(re.findall(
        r"<plugin name='([^']*)' filename='([^']*)'", sdf))
    assert plugins.get('gas_sensor_plugin') == 'libgas_sensor_plugin.so'
    assert re.search(
        r'<sensor_link_name>\s*gas_sensor_link\s*</sensor_link_name>', sdf)
    # Identity map->odom is only valid against world-referenced odometry.
    assert re.search(r'<odometry_source>\s*1\s*</odometry_source>', sdf)
    joint = re.search(
        r"<joint name='gas_sensor_link_fixed'.*?</joint>", sdf, re.S).group(0)
    pose = re.search(r"<pose relative_to='base_footprint'>([^<]*)<", joint)
    # On the rotation axis: turning in place must not move the sensor in
    # (x, y), so a measurement belongs to the cell the robot stands in.
    assert [float(v) for v in pose.group(1).split()[:3]] == [0.0, 0.0, 0.45]


def test_amc_sdf_matches_regeneration_from_the_urdf():
    """The SDF is generated, so a stale copy must fail rather than drift."""
    gz = subprocess.run(
        ['gz', 'sdf', '-p', 'mobile_amc.urdf'], cwd=PACKAGE/'urdf',
        capture_output=True, text=True, check=False)
    if gz.returncode != 0 or not gz.stdout.strip():
        pytest.skip('gz sdf is unavailable in this environment')
    expected = (PACKAGE/'urdf'/'mobile_amc.sdf').read_text()
    assert gz.stdout.split() == expected.split()


def test_render_robot_sdf_applies_motion_to_both_robots():
    motion = {'max_wheel_acceleration': 1.25, 'lidar_update_rate': 9.0}
    for model, sensor in (
            ('tb3_with_gas_sensor.sdf', None),
            ('mobile_amc.sdf', 'amc_lidar')):
        xml = (PACKAGE/'urdf'/model).read_text()
        rendered = render_robot_sdf(xml, motion, sensor)
        assert '<max_wheel_acceleration>1.25</max_wheel_acceleration>' in rendered
        name = sensor or DEFAULT_ROBOT['lidar_sensor_name']
        block = re.search(
            r'<sensor\s+name=[\'"]' + name + r'[\'"].*?<update_rate>([^<]*)<',
            rendered, re.S)
        assert float(block.group(1)) == 9.0


def test_render_robot_sdf_rejects_a_sensor_name_the_model_lacks():
    xml = (PACKAGE/'urdf'/'mobile_amc.sdf').read_text()
    motion = {'max_wheel_acceleration': 1.0, 'lidar_update_rate': 5.0}
    with pytest.raises(ValueError, match='hls_lfcd_lds'):
        render_robot_sdf(xml, motion)


def test_gmrf_field_resolution_matches_the_profile(profile):
    gas = profile['gas_environment']
    field = field_geometry(
        gas['map_min_x'], gas['map_max_x'],
        gas['map_min_y'], gas['map_max_y'], gas['gmrf_resolution'])
    assert gas['gmrf_resolution'] == 0.5
    assert (field.width, field.height) == (41, 29)
