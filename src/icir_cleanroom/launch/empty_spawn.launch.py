import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('icir_cleanroom')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')

    world_file = os.path.join(pkg_dir, 'worlds', 'empty.world')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'tb3_with_gas_sensor.urdf')
    sdf_file = os.path.join(pkg_dir, 'urdf', 'tb3_with_gas_sensor.sdf')
    rviz_config_file = os.path.join(pkg_dir, 'rviz', 'default.rviz')

    os.environ['TURTLEBOT3_MODEL'] = 'waffle_pi'

    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file, 'verbose': 'true'}.items()
    )

    gazebo_client = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
        additional_env={
            'LIBGL_ALWAYS_SOFTWARE': '1'},
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': open(urdf_file, 'r').read()
        }]
    )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', sdf_file,
            '-entity', 'turtlebot3_with_gas_sensor',
            '-x', '-5.0',
            '-y', '-2.0',
            '-z', '0.01'
        ],
        output='screen'
    )

    map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        gazebo_server,
        gazebo_client,
        robot_state_publisher,
        spawn_robot,
        map_to_odom_tf,
        rviz,
    ])

