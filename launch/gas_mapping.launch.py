import atexit
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    OpaqueFunction, SetEnvironmentVariable, TimerAction)
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from icir_cleanroom.gas_mapping.navigation_profile import (
    load_navigation_settings, navigation_goal_clearance, render_robot_sdf)


def remove_temporary_file(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def launch_setup(context):
    package_dir = get_package_share_directory('icir_cleanroom')
    gazebo_dir = get_package_share_directory('gazebo_ros')
    environment_name = LaunchConfiguration('environment').perform(context)
    profile_path = os.path.join(
        package_dir, 'config', 'environments', f'{environment_name}.yaml')
    if not os.path.isfile(profile_path):
        raise RuntimeError(f'Environment profile not found: {profile_path}')
    profile = load_yaml(profile_path)
    environment = profile['environment']
    world = os.path.join(package_dir, environment['world'])
    map_yaml = os.path.join(package_dir, environment['map'])
    navigation_profile = environment.get('navigation_profile')
    if not isinstance(navigation_profile, str) or not navigation_profile:
        raise RuntimeError(
            f'Environment navigation_profile is required: {profile_path}')
    navigation_profile_path = os.path.join(
        package_dir, navigation_profile)
    for description, path in (
            ('world', world), ('map', map_yaml),
            ('navigation profile', navigation_profile_path)):
        if not os.path.isfile(path):
            raise RuntimeError(f'Environment {description} not found: {path}')

    robot_urdf = os.path.join(
        package_dir, 'urdf', 'tb3_with_gas_sensor.urdf')
    robot_sdf = os.path.join(
        package_dir, 'urdf', 'tb3_with_gas_sensor.sdf')
    rviz = os.path.join(package_dir, 'rviz', 'cleanroom_empty.rviz')
    base_nav2_params = os.path.join(
        package_dir, 'config', 'nav2_params.yaml')
    try:
        nav2_config, robot_motion = load_navigation_settings(
            base_nav2_params, navigation_profile_path)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise RuntimeError(
            f'Invalid navigation profile {navigation_profile_path}: '
            f'{error}') from error
    with tempfile.NamedTemporaryFile(
            mode='w', prefix='icir_navigation_', suffix='.yaml',
            delete=False, encoding='utf-8') as stream:
        yaml.safe_dump(nav2_config, stream, sort_keys=False)
        temporary_nav2_params = stream.name
    atexit.register(remove_temporary_file, temporary_nav2_params)
    mapping_params = load_yaml(os.path.join(
        package_dir, 'config', 'mapping', 'default.yaml'))[
            'gas_mapping_controller_node']['ros__parameters']

    with open(robot_urdf, 'r', encoding='utf-8') as stream:
        robot_description = stream.read()
    with open(robot_sdf, 'r', encoding='utf-8') as stream:
        robot_xml = stream.read()
    try:
        robot_xml = render_robot_sdf(robot_xml, robot_motion)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f'Invalid robot motion profile: {error}') from error
    with tempfile.NamedTemporaryFile(
            mode='w', prefix='icir_mapping_robot_', suffix='.sdf',
            delete=False, encoding='utf-8') as stream:
        stream.write(robot_xml)
        temporary_sdf = stream.name
    atexit.register(remove_temporary_file, temporary_sdf)

    spawn = environment['robot_spawn']
    use_sim_time = LaunchConfiguration('use_sim_time')
    headless = LaunchConfiguration('headless')
    gas_parameters = dict(profile['gas_environment'])
    gas_parameters['use_sim_time'] = use_sim_time
    try:
        gas_parameters['navigation_goal_clearance'] = (
            navigation_goal_clearance(nav2_config))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f'Invalid Nav2 inflation configuration: {error}') from error
    lrs_parameters = dict(profile['lrs'])
    required_lrs_parameters = (
        'lrs_cluster_count', 'lrs_cluster_random_seed')
    missing_lrs_parameters = [
        name for name in required_lrs_parameters
        if name not in lrs_parameters]
    if missing_lrs_parameters:
        raise RuntimeError(
            f'Environment LRS profile is missing required values: '
            f'{", ".join(missing_lrs_parameters)}')
    lrs_parameters['use_sim_time'] = use_sim_time
    gas_parameters.update({
        'lrs_cluster_count': lrs_parameters['lrs_cluster_count'],
        'lrs_cluster_random_seed':
            lrs_parameters['lrs_cluster_random_seed'],
    })
    controller_parameters = dict(mapping_params)
    controller_parameters['use_sim_time'] = use_sim_time
    model_path = os.path.join(
        package_dir, 'models', 'aws_small_warehouse')
    gazebo_model_path = model_path
    if os.environ.get('GAZEBO_MODEL_PATH'):
        gazebo_model_path += os.pathsep + os.environ['GAZEBO_MODEL_PATH']

    actions = [
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                gazebo_dir, 'launch', 'gzserver.launch.py')),
            launch_arguments={'world': world}.items()),
        ExecuteProcess(
            cmd=['gzclient'], output='screen',
            condition=UnlessCondition(headless)),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_description}],
            output='screen'),
        Node(
            package='gazebo_ros', executable='spawn_entity.py',
            arguments=[
                '-file', temporary_sdf,
                '-entity', 'turtlebot3_with_gas_sensor',
                '-x', str(spawn['x']), '-y', str(spawn['y']),
                '-z', str(spawn['z']), '-Y', str(spawn['yaw'])],
            output='screen'),
        Node(
            package='tf2_ros', executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen'),
        Node(
            package='nav2_map_server', executable='map_server',
            name='map_server',
            parameters=[{'use_sim_time': use_sim_time,
                         'yaml_filename': map_yaml}], output='screen'),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map_server',
            parameters=[{'use_sim_time': use_sim_time,
                         'autostart': True,
                         'node_names': ['map_server']}], output='screen'),
        Node(
            package='icir_cleanroom', executable='gas_environment_node.py',
            parameters=[gas_parameters], output='screen'),
        Node(
            package='icir_cleanroom', executable='lrs_path_planner_node.py',
            parameters=[lrs_parameters], output='screen'),
        TimerAction(period=3.0, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('nav2_bringup'),
                'launch', 'navigation_launch.py')),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file': temporary_nav2_params,
                'autostart': 'true'}.items())]),
        TimerAction(period=6.0, actions=[Node(
            package='icir_cleanroom',
            executable='gas_mapping_controller_node.py',
            parameters=[controller_parameters], output='screen')]),
        Node(
            package='rviz2', executable='rviz2',
            arguments=['-d', rviz], output='screen',
            condition=UnlessCondition(headless)),
    ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('environment', default_value='empty_50m'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('headless', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])
