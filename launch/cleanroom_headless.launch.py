"""
Headless(무GUI) 클린룸 GSL 실행 - 모바일/SSH에서 명령어만으로 데이터 수집용.

gzclient(Gazebo GUI)와 rviz를 띄우지 않아 화면 없이도 동작한다.
use_grid_refine 인자로 두 실험 버전을 전환한다:

  # SPIRAL만 사용 (baseline)
  ros2 launch icir_cleanroom cleanroom_headless.launch.py use_grid_refine:=false

  # Kernel DM + SPIRAL 통합 (나선 수렴 후 그리드 argmax로 보정 이동)
  ros2 launch icir_cleanroom cleanroom_headless.launch.py use_grid_refine:=true
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg_dir = get_package_share_directory('icir_cleanroom')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')

    world_file = os.path.join(pkg_dir, 'worlds', 'cleanroom.world')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'tb3_with_gas_sensor.urdf')
    sdf_file = os.path.join(pkg_dir, 'urdf', 'tb3_with_gas_sensor.sdf')

    nav2_params_path = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
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

    use_grid_refine = LaunchConfiguration('use_grid_refine')

    # gzserver만 (물리 + 센서 플러그인 headless). gzclient/rviz 없음.
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file, 'verbose': 'true'}.items()
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
        parameters=[{
            'use_sim_time': True,
            'use_grid_refine': ParameterValue(use_grid_refine, value_type=bool),
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'use_grid_refine', default_value='true',
            description='true=Kernel DM+SPIRAL 통합, false=SPIRAL만(baseline)'),
        gazebo_server,
        robot_state_publisher,
        spawn_robot,
        map_to_odom_tf,
        TimerAction(period=5.0, actions=[nav2_bringup_launch]),
        TimerAction(period=8.0, actions=[gas_patrol_node]),
    ])
