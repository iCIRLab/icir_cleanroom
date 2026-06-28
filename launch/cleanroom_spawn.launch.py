import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    pkg_dir = get_package_share_directory('icir_cleanroom')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')

    world_file = os.path.join(pkg_dir, 'worlds', 'cleanroom.world')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'tb3_with_gas_sensor.urdf')
    sdf_file = os.path.join(pkg_dir, 'urdf', 'tb3_with_gas_sensor.sdf')
    rviz_config_file = os.path.join(pkg_dir, 'rviz', 'default.rviz')

    nav2_params_path = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    # TODO: SLAM으로 클린룸 맵 생성 후 cleanroom.yaml / cleanroom.pgm 으로 교체
    map_yaml_path = os.path.join(pkg_dir, 'map', 'cleanroom.yaml')
    map_data_yaml_path = os.path.join(pkg_dir, 'config', 'gas_attraction', 'map_data.yaml')
    idw_yaml_path = os.path.join(pkg_dir, 'config', 'gas_attraction', 'idw.yaml')

    configured_params = RewrittenYaml(
        source_file=nav2_params_path,
        root_key='',
        param_rewrites={
            'map_data_yaml': map_data_yaml_path,
            'idw_yaml': idw_yaml_path,
        },
        convert_types=True
    )

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
        additional_env={'LIBGL_ALWAYS_SOFTWARE': '1'},
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

    # 로봇 초기 위치: 중앙 복도 하단
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', sdf_file,
            '-entity', 'turtlebot3_with_gas_sensor',
            '-x', '0.0',
            '-y', '-4.0',
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

    gas_concentration_marker = Node(
        package='icir_cleanroom',
        executable='gas_concentration_marker_node.py',
        name='gas_concentration_marker_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('nav2_bringup'),
                'launch',
                'bringup_launch.py'
            )
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'map': map_yaml_path,
            'params_file': configured_params
        }.items()
    )

    gas_patrol_node = Node(
        package='icir_cleanroom',
        executable='gas_patrol_node.py',
        name='gas_patrol_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        gazebo_server,
        gazebo_client,
        robot_state_publisher,
        spawn_robot,
        map_to_odom_tf,
        rviz,
        gas_concentration_marker,
        TimerAction(period=5.0, actions=[nav2_bringup_launch]),
        TimerAction(period=8.0, actions=[gas_patrol_node]),
    ])
~ㅇ