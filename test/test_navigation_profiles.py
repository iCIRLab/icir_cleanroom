"""Characterize environment-specific Nav2 and robot motion settings."""

from pathlib import Path
import xml.etree.ElementTree as element_tree

import pytest
import yaml

from icir_cleanroom.gas_mapping.navigation_profile import (
    load_navigation_settings, navigation_goal_clearance, recursive_merge,
    render_robot_sdf, validated_robot_motion)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BASE_PARAMS = PACKAGE_ROOT / 'config' / 'nav2_params.yaml'
ROBOT_SDF = PACKAGE_ROOT / 'urdf' / 'tb3_with_gas_sensor.sdf'


def load_profile(name):
    return load_navigation_settings(
        BASE_PARAMS,
        PACKAGE_ROOT / 'config' / 'navigation' / f'{name}.yaml')


def follow_path(config):
    return config['controller_server']['ros__parameters']['FollowPath']


def test_recursive_merge_preserves_nested_common_values_without_mutation():
    base = {'node': {'ros__parameters': {'common': 1, 'selected': 2}}}
    overrides = {'node': {'ros__parameters': {'selected': 3}}}

    merged = recursive_merge(base, overrides)

    assert merged == {
        'node': {'ros__parameters': {'common': 1, 'selected': 3}}}
    assert base['node']['ros__parameters']['selected'] == 2
    assert overrides['node']['ros__parameters']['selected'] == 3


def test_empty_fast_profile_preserves_existing_fast_motion():
    config, motion = load_profile('empty_fast')

    assert motion == {
        'max_wheel_acceleration': 152.0,
        'lidar_update_rate': 5.0,
    }
    assert follow_path(config)['max_vel_x'] == 1.0
    assert follow_path(config)['max_speed_xy'] == 1.0
    assert follow_path(config)['acc_lim_x'] == 5.0
    assert follow_path(config)['decel_lim_x'] == -5.0


def test_aws_profile_applies_safe_motion_and_keeps_common_nav2_settings():
    config, motion = load_profile('aws_warehouse_safe')
    controller = config['controller_server']['ros__parameters']
    path = controller['FollowPath']
    local = config['local_costmap']['local_costmap']['ros__parameters']
    global_params = config['global_costmap']['global_costmap'][
        'ros__parameters']

    assert motion == {
        'max_wheel_acceleration': 24.0,
        'lidar_update_rate': 10.0,
    }
    assert controller['controller_frequency'] == 15.0
    assert path['max_vel_x'] == 0.6
    assert path['max_speed_xy'] == 0.4
    assert path['max_vel_theta'] == 1.8
    assert path['acc_lim_x'] == 1.0
    assert path['decel_lim_x'] == -1.0
    assert path['vtheta_samples'] == 30
    assert path['sim_time'] == 1.7
    assert path['critics'] == [
        'RotateToGoal', 'Oscillation', 'BaseObstacle', 'GoalAlign',
        'PathAlign', 'PathDist', 'GoalDist']
    assert local['update_frequency'] == 5.0
    assert local['width'] == 3
    assert local['height'] == 3
    assert local['resolution'] == 0.05
    assert local['robot_radius'] == 0.22
    assert local['plugins'] == [
        'static_layer', 'voxel_layer', 'inflation_layer']
    assert local['inflation_layer']['inflation_radius'] == 0.45
    assert local['inflation_layer']['cost_scaling_factor'] == 5.0
    assert global_params['update_frequency'] == 1.0
    assert global_params['inflation_layer']['inflation_radius'] == 0.45
    assert global_params['inflation_layer']['cost_scaling_factor'] == 5.0
    assert navigation_goal_clearance(config) == 0.45


def test_aws_profile_enables_dwb_diagnostics_only_for_warehouse():
    aws_config, _ = load_profile('aws_warehouse_safe')
    empty_config, _ = load_profile('empty_fast')
    aws_path = follow_path(aws_config)
    empty_path = follow_path(empty_config)

    assert aws_path['publish_evaluation'] is True
    assert aws_path['publish_trajectories'] is True
    assert aws_path['publish_cost_grid_pc'] is True
    assert 'publish_evaluation' not in empty_path
    assert 'publish_trajectories' not in empty_path
    assert 'publish_cost_grid_pc' not in empty_path


def test_empty_profile_keeps_local_costmap_plugins_and_clearance():
    config, _ = load_profile('empty_fast')
    local = config['local_costmap']['local_costmap']['ros__parameters']

    assert local['plugins'] == ['voxel_layer', 'inflation_layer']
    assert navigation_goal_clearance(config) == 0.25


@pytest.mark.parametrize('radius', [None, 0.0, -0.1, float('nan')])
def test_navigation_goal_clearance_rejects_missing_or_invalid_radius(radius):
    config, _ = load_profile('aws_warehouse_safe')
    inflation = config['local_costmap']['local_costmap'][
        'ros__parameters']['inflation_layer']
    if radius is None:
        del inflation['inflation_radius']
    else:
        inflation['inflation_radius'] = radius

    with pytest.raises(ValueError, match='inflation_radius'):
        navigation_goal_clearance(config)


@pytest.mark.parametrize(
    ('name', 'max_velocity', 'max_accel'),
    [('empty_fast', [1.0, 0.0, 3.0], [5.0, 0.0, 3.2]),
     ('aws_warehouse_safe', [0.4, 0.0, 1.8], [0.8, 0.0, 2.0])])
def test_velocity_smoother_matches_selected_controller_profile(
        name, max_velocity, max_accel):
    config, _ = load_profile(name)
    smoother = config['velocity_smoother']['ros__parameters']

    assert smoother['max_velocity'] == max_velocity
    assert smoother['max_accel'] == max_accel


def test_robot_sdf_motion_values_are_profile_driven():
    source = ROBOT_SDF.read_text(encoding='utf-8')

    empty_xml = render_robot_sdf(source, load_profile('empty_fast')[1])
    aws_xml = render_robot_sdf(
        source, load_profile('aws_warehouse_safe')[1])

    assert '<max_wheel_acceleration>152.0</max_wheel_acceleration>' in empty_xml
    assert '<max_wheel_acceleration>24.0</max_wheel_acceleration>' in aws_xml
    empty_lidar = empty_xml.split(
        '<sensor name="hls_lfcd_lds"', 1)[1].split('</sensor>', 1)[0]
    aws_lidar = aws_xml.split(
        '<sensor name="hls_lfcd_lds"', 1)[1].split('</sensor>', 1)[0]
    assert '<update_rate>5.0</update_rate>' in empty_lidar
    assert '<update_rate>10.0</update_rate>' in aws_lidar
    assert '<update_rate>200</update_rate>' in aws_xml


def test_gas_sensor_update_rate_is_explicitly_limited():
    root = element_tree.fromstring(
        ROBOT_SDF.read_text(encoding='utf-8'))
    plugin = root.find(".//plugin[@name='gas_sensor_plugin']")

    assert plugin is not None
    assert float(plugin.findtext('update_rate')) == 20.0


@pytest.mark.parametrize(
    'motion',
    [None, {}, {'max_wheel_acceleration': 24.0},
     {'max_wheel_acceleration': 0.0, 'lidar_update_rate': 10.0}])
def test_robot_motion_requires_all_positive_values(motion):
    with pytest.raises(ValueError):
        validated_robot_motion(motion)


def test_navigation_profile_requires_a_real_mapping_file(tmp_path):
    invalid_profile = tmp_path / 'invalid.yaml'
    invalid_profile.write_text('- not\n- a\n- mapping\n', encoding='utf-8')

    with pytest.raises(ValueError, match='must contain a YAML mapping'):
        load_navigation_settings(BASE_PARAMS, invalid_profile)
    with pytest.raises(ValueError, match='not found'):
        load_navigation_settings(BASE_PARAMS, tmp_path / 'missing.yaml')


def test_navigation_profile_requires_nav2_overrides(tmp_path):
    motion_only = tmp_path / 'motion_only.yaml'
    motion_only.write_text(yaml.safe_dump({'robot_motion': {
        'max_wheel_acceleration': 24.0,
        'lidar_update_rate': 10.0,
    }}), encoding='utf-8')

    with pytest.raises(ValueError, match='has no Nav2 overrides'):
        load_navigation_settings(BASE_PARAMS, motion_only)
