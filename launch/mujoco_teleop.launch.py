"""Drive a robot in a MuJoCo cleanroom by gamepad or keyboard.

ros2 launch icir_cleanroom mujoco_teleop.launch.py \
    world:=cleanroom_amc robot:=mobile_amc joy:=true
Without joy:=true, drive from another terminal with
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from icir_cleanroom.mujoco_sim.scene import load_robot_config


def launch_setup(context):
    package_dir = get_package_share_directory('icir_cleanroom')
    world = LaunchConfiguration('world').perform(context)
    robot_name = LaunchConfiguration('robot').perform(context)
    try:
        robot = load_robot_config(
            os.path.join(package_dir, 'config', 'mujoco', 'robots.yaml'),
            robot_name)
    except ValueError as error:
        raise RuntimeError(str(error)) from error

    def flag(name):
        return LaunchConfiguration(name).perform(context).lower() == 'true'

    actions = [
        Node(
            package='icir_cleanroom', executable='mujoco_bridge_node.py',
            parameters=[{
                'world': world, 'robot': robot_name,
                'x': float(LaunchConfiguration('x').perform(context)),
                'y': float(LaunchConfiguration('y').perform(context)),
                'yaw': float(LaunchConfiguration('yaw').perform(context)),
                'viewer': flag('viewer')}],
            output='screen'),
        # The bridge publishes ground-truth odometry, as the Gazebo
        # diff drive does with <odometry_source>1</odometry_source>.
        Node(
            package='tf2_ros', executable='static_transform_publisher',
            arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
            output='screen'),
    ]
    if robot.get('description'):
        with open(os.path.join(package_dir, robot['description']), 'r',
                  encoding='utf-8') as stream:
            description = stream.read()
        actions.append(Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'use_sim_time': True,
                         'robot_description': description}],
            output='screen'))
    maps = sorted((Path(package_dir) / 'maps' / world).glob('*.yaml'))
    if maps:
        actions += [
            Node(
                package='nav2_map_server', executable='map_server',
                name='map_server',
                parameters=[{'use_sim_time': True,
                             'yaml_filename': str(maps[0])}],
                output='screen'),
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_map_server',
                parameters=[{'use_sim_time': True, 'autostart': True,
                             'node_names': ['map_server']}],
                output='screen'),
        ]
    if flag('joy'):
        joy = os.path.join(package_dir, 'config', 'teleop', 'joy.yaml')
        actions += [
            Node(package='joy', executable='joy_node', name='joy_node',
                 parameters=[joy], output='screen'),
            Node(package='teleop_twist_joy', executable='teleop_node',
                 name='teleop_twist_joy_node', parameters=[joy],
                 output='screen'),
        ]
    else:
        actions.append(LogInfo(msg=(
            'Keyboard driving: ros2 run teleop_twist_keyboard '
            'teleop_twist_keyboard')))
    if flag('rviz'):
        actions.append(Node(
            package='rviz2', executable='rviz2',
            arguments=['-d', os.path.join(
                package_dir, 'rviz', 'mujoco_teleop.rviz')],
            parameters=[{'use_sim_time': True}], output='screen'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value='cleanroom_amc',
            description='Folder under worlds/mujoco/'),
        DeclareLaunchArgument(
            'robot', default_value='mobile_amc',
            description='Entry in config/mujoco/robots.yaml'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('viewer', default_value='true',
                              description='Open the MuJoCo viewer'),
        DeclareLaunchArgument('joy', default_value='false',
                              description='Drive with a gamepad'),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])
