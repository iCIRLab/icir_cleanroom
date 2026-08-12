import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('icir_cleanroom')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')

    world_file = os.path.join(pkg_dir, 'worlds', 'cleanroom.world')
    map_file = os.path.join(pkg_dir, 'map', 'cleanroom.yaml')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'tb3_with_gas_sensor.urdf')
    sdf_file = os.path.join(pkg_dir, 'urdf', 'tb3_with_gas_sensor.sdf')
    rviz_file = os.path.join(pkg_dir, 'rviz', 'cleanroom_view.rviz')
    nav2_params_file = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')

    with open(urdf_file, 'r', encoding='utf-8') as urdf:
        robot_description = urdf.read()

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_dir, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world_file}.items(),
        ),
        ExecuteProcess(
            cmd=['gzclient'],
            output='screen',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'use_sim_time': True,
                'robot_description': robot_description,
            }],
            output='screen',
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-file', sdf_file,
                '-entity', 'turtlebot3_with_gas_sensor',
                '-x', '0.0', '-y', '-4.0', '-z', '0.01',
            ],
            output='screen',
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen',
        ),
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            parameters=[{
                'yaml_filename': map_file,
                'use_sim_time': True,
            }],
            output='screen',
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            parameters=[{
                'use_sim_time': True,
                'autostart': True,
                'node_names': ['map_server'],
            }],
            output='screen',
        ),
        Node(
            package='icir_cleanroom',
            executable='gas_grid_map_node.py',
            parameters=[{
                'use_sim_time': True,
                'grid_resolution': 0.5,
                'grid_margin': 0.5,
                'dfs_start_x': 0.0,
                'dfs_start_y': -4.0,
            }],
            output='screen',
        ),
        TimerAction(
            period=3.0,
            actions=[IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('nav2_bringup'),
                        'launch',
                        'navigation_launch.py',
                    )
                ),
                launch_arguments={
                    'use_sim_time': 'true',
                    'params_file': nav2_params_file,
                    'autostart': 'true',
                }.items(),
            )],
        ),
        TimerAction(
            period=6.0,
            actions=[Node(
                package='icir_cleanroom',
                executable='gas_grid_sampler_node.py',
                parameters=[{
                    'use_sim_time': True,
                    'dwell_seconds': 5.0,
                    'max_retries': 2,
                }],
                output='screen',
            )],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_file],
            output='screen',
        ),
    ])
