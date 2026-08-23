"""Environment-specific Nav2 and simulated robot motion profiles."""

import copy
import math
import os
import re

import yaml


ROBOT_MOTION_KEYS = (
    'max_wheel_acceleration',
    'lidar_update_rate',
)


def recursive_merge(base, overrides):
    """Return a deep merge without mutating either input mapping."""
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        raise TypeError('navigation base and overrides must be mappings')
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = recursive_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml_mapping(path, description):
    if not os.path.isfile(path):
        raise ValueError(f'{description} not found: {path}')
    with open(path, 'r', encoding='utf-8') as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f'{description} must contain a YAML mapping: {path}')
    return loaded


def validated_robot_motion(value):
    if not isinstance(value, dict):
        raise ValueError('navigation profile requires robot_motion mapping')
    missing = [key for key in ROBOT_MOTION_KEYS if key not in value]
    if missing:
        raise ValueError(
            'navigation profile robot_motion is missing: '
            + ', '.join(missing))
    result = {}
    for key in ROBOT_MOTION_KEYS:
        number = float(value[key])
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f'robot_motion.{key} must be finite and positive')
        result[key] = number
    return result


def load_navigation_settings(base_path, profile_path):
    """Load, validate, and merge a base Nav2 file and partial profile."""
    base = load_yaml_mapping(base_path, 'Base Nav2 parameters')
    profile = load_yaml_mapping(profile_path, 'Navigation profile')
    motion = validated_robot_motion(profile.get('robot_motion'))
    overrides = {
        key: value for key, value in profile.items()
        if key != 'robot_motion'}
    if not overrides:
        raise ValueError('navigation profile has no Nav2 overrides')
    return recursive_merge(base, overrides), motion


def navigation_goal_clearance(nav2_config):
    """Return the largest effective Nav2 inflation radius."""
    if not isinstance(nav2_config, dict):
        raise ValueError('merged Nav2 parameters must be a mapping')
    paths = (
        ('local_costmap', 'local_costmap'),
        ('global_costmap', 'global_costmap'),
    )
    radii = []
    for node_name, costmap_name in paths:
        try:
            value = nav2_config[node_name][costmap_name][
                'ros__parameters']['inflation_layer']['inflation_radius']
        except (KeyError, TypeError) as error:
            raise ValueError(
                f'{node_name}.{costmap_name} inflation_radius is required') \
                from error
        radius = float(value)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(
                f'{node_name}.{costmap_name} inflation_radius must be '
                'finite and positive')
        radii.append(radius)
    return max(radii)


def replace_unique(xml, pattern, value, description):
    matches = re.findall(pattern, xml, flags=re.DOTALL)
    if len(matches) != 1:
        raise ValueError(
            f'expected one {description} in robot SDF, found {len(matches)}')
    return re.sub(
        pattern, lambda match: f'{match.group(1)}{value}{match.group(2)}',
        xml, count=1, flags=re.DOTALL)


def render_robot_sdf(robot_xml, robot_motion):
    """Apply validated profile motion values to the temporary robot SDF."""
    motion = validated_robot_motion(robot_motion)
    output = replace_unique(
        robot_xml,
        r'(<max_wheel_acceleration>)\s*[^<]+\s*'
        r'(</max_wheel_acceleration>)',
        str(motion['max_wheel_acceleration']),
        'max_wheel_acceleration element')
    return replace_unique(
        output,
        r'(<sensor\s+name=["\']hls_lfcd_lds["\'][^>]*>.*?'
        r'<update_rate>)\s*[^<]+\s*(</update_rate>)',
        str(motion['lidar_update_rate']),
        'hls_lfcd_lds update_rate element')


__all__ = [
    'load_navigation_settings', 'navigation_goal_clearance',
    'recursive_merge', 'render_robot_sdf', 'validated_robot_motion']
