import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('icir_cleanroom')
    gazebo_dir = get_package_share_directory('gazebo_ros')
    world = os.path.join(pkg_dir, 'worlds', 'empty_lrs_50m.world')
    urdf = os.path.join(pkg_dir, 'urdf', 'tb3_with_gas_sensor.urdf')
    sdf = os.path.join(pkg_dir, 'urdf', 'tb3_with_gas_sensor.sdf')
    rviz = os.path.join(pkg_dir, 'rviz', 'cleanroom_empty.rviz')
    nav2_params = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    with open(urdf, 'r', encoding='utf-8') as stream:
        robot_description = stream.read()

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_dir, 'launch', 'gzserver.launch.py')),
            launch_arguments={'world': world}.items()),
        ExecuteProcess(cmd=['gzclient'], output='screen'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'use_sim_time': True, 'robot_description': robot_description}],
             output='screen'),
        Node(package='gazebo_ros', executable='spawn_entity.py',
             arguments=['-file', sdf, '-entity', 'turtlebot3_with_gas_sensor',
                        '-x', '0.0', '-y', '0.0', '-z', '0.01'], output='screen'),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'], output='screen'),
        Node(package='icir_cleanroom', executable='gas_empty_map_node.py',
             parameters=[{
                 'use_sim_time': True,
                 'source_x': 10.0, 'source_y': 10.0,
                 'source_strength': 1.0, 'source_sigma': 5.0,
                 'map_min_x': 0.0, 'map_max_x': 50.0,
                 'map_min_y': 0.0, 'map_max_y': 50.0,
                 'nav_resolution': 0.05,
                 'ground_truth_resolution': 0.1,
                 'gmrf_resolution': 1.0,
                 'log_concentration_min': -4.0,
                 'map_margin': 2.5,
             }], output='screen'),
        Node(package='icir_cleanroom', executable='lrs_path_planner_node.py',
             parameters=[{
                 'use_sim_time': True,
                 # Paper spatial setup: d_res=1.0m, alpha=10.
                 'lrs_spacing': 10.0,
                 'lrs_min_x': 0.0, 'lrs_max_x': 50.0,
                 'lrs_min_y': 0.0, 'lrs_max_y': 50.0,
                 'robot_start_x': 0.0, 'robot_start_y': 0.0,
                 'goal_clearance': 0.25,
                 'tsp_time_limit': 60.0,
             }], output='screen'),
        TimerAction(period=3.0, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')),
            launch_arguments={'use_sim_time': 'true', 'params_file': nav2_params,
                              'autostart': 'true'}.items())]),
        TimerAction(period=6.0, actions=[Node(
            package='icir_cleanroom', executable='gas_mapping_controller_node.py',
            parameters=[{
                'use_sim_time': True, 'dwell_seconds': 1.0, 'max_retries': 2,
                'observation_variance_scale': 0.01,
                'observation_variance_floor': 0.001,
                'smoothness_precision': 1.0,
                # Preserve total smoothness while increasing diagonal coupling.
                'cardinal_weight': 0.75, 'diagonal_weight': 0.25,
                'background_precision': 1.0, 'background_mean': 0.0,
                'estimate_resolution': 0.1,
                'log_concentration_min': -4.0,
            }], output='screen')]),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz], output='screen'),
    ])
