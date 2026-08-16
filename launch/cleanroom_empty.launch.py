import atexit
import math
import os
import tempfile
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def remove_temporary_file(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def generate_launch_description():
    pkg_dir = get_package_share_directory('icir_cleanroom')
    gazebo_dir = get_package_share_directory('gazebo_ros')
    world = os.path.join(pkg_dir, 'worlds', 'empty_lrs_50m.world')
    urdf = os.path.join(pkg_dir, 'urdf', 'tb3_with_gas_sensor.urdf')
    base_sdf = os.path.join(pkg_dir, 'urdf', 'tb3_with_gas_sensor.sdf')
    rviz = os.path.join(pkg_dir, 'rviz', 'cleanroom_empty.rviz')
    nav2_params = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    with open(urdf, 'r', encoding='utf-8') as stream:
        robot_description = stream.read()
    with open(base_sdf, 'r', encoding='utf-8') as stream:
        empty_robot_sdf = stream.read()
    wheel_limit = '<max_wheel_acceleration>1.0</max_wheel_acceleration>'
    if empty_robot_sdf.count(wheel_limit) != 1:
        raise RuntimeError('Expected one wheel acceleration limit in robot SDF')
    empty_robot_sdf = empty_robot_sdf.replace(
        wheel_limit,
        '<max_wheel_acceleration>152.0</max_wheel_acceleration>')
    with tempfile.NamedTemporaryFile(
            mode='w', prefix='icir_empty_mapping_robot_', suffix='.sdf',
            delete=False, encoding='utf-8') as stream:
        stream.write(empty_robot_sdf)
        empty_sdf = stream.name
    atexit.register(remove_temporary_file, empty_sdf)

    # Keep the paper's 5 m/s value in the HRS time-budget model below, while
    # using a stable physical Nav2 speed for the simulated TurtleBot. Gas
    # sampling depends on position, so the final yaw is intentionally ignored.
    with open(nav2_params, 'r', encoding='utf-8') as stream:
        empty_nav2_config = yaml.safe_load(stream)

    # 경로 추정 컨트롤러 파라미터
    follow_path = empty_nav2_config['controller_server']['ros__parameters']['FollowPath']
    follow_path['max_vel_x'] = 1.0          # 최대 전진 선속도 (m/s)
    follow_path['max_speed_xy'] = 1.0       # 최대 합성 속도 (전후진 + 좌우 이동 속도 한계 m/s)
    follow_path['acc_lim_x'] = 5.0          # 전진 방향 최대 가속도 (m/s²)
    follow_path['decel_lim_x'] = -5.0       # 전진 방향 최대 감속도 (m/s²)
    follow_path['xy_goal_tolerance'] = 0.15 # 몇 미터 이내로 들어오면 "도착"으로 판단

    # 최종 목표 도달 판정기
    goal_checker = empty_nav2_config['controller_server']['ros__parameters']['general_goal_checker']
    goal_checker['xy_goal_tolerance'] = 0.15        # 최종 목표 지점까지 위치 허용 오차 
    goal_checker['yaw_goal_tolerance'] = math.pi    # 최종 목표 지점에서 방향 허용 오차 (180도 = 방향 고려 x)

    # 속도 명령 스무딩 필터
    velocity_smoother = empty_nav2_config['velocity_smoother']['ros__parameters']
    velocity_smoother['max_velocity'][0] = 1.0      # x축(전진) 방향 최대 속도 한계 (m/s)
    velocity_smoother['max_accel'][0] = 5.0         # x축 방향 최대 가속도 한계 (m/s²)
    velocity_smoother['max_decel'][0] = -5.0        # x축 방향 최대 감속도 한계 (m/s²)

    with tempfile.NamedTemporaryFile(
            mode='w', prefix='icir_empty_mapping_nav2_', suffix='.yaml',
            delete=False, encoding='utf-8') as stream:
        yaml.safe_dump(empty_nav2_config, stream, sort_keys=False)
        empty_nav2_params = stream.name
    atexit.register(remove_temporary_file, empty_nav2_params)

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_dir, 'launch', 'gzserver.launch.py')),
            launch_arguments={'world': world}.items()),
        ExecuteProcess(cmd=['gzclient'], output='screen'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'use_sim_time': True, 'robot_description': robot_description}],
             output='screen'),
        Node(package='gazebo_ros', executable='spawn_entity.py',
             arguments=['-file', empty_sdf,
                        '-entity', 'turtlebot3_with_gas_sensor',
                        '-x', '0.0', '-y', '0.0', '-z', '0.01'], output='screen'),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'], output='screen'),
        Node(package='icir_cleanroom', executable='gas_empty_map_node.py',
             parameters=[{
                 'use_sim_time': True,
                 # Repeated hotspot scenario for history-aware LRS validation.
                 # The controller never receives these hidden GT parameters.
                 'source_mode': 'recurrent_hotspots_after_peak',
                 'source_enabled': False,
                 'source_x': 10.0, 'source_y': 10.0,
                 'source_strength': 1.0, 'source_sigma': 5.0,
                 'source_random_seed': -1,
                 'source_random_sigma_min': 4.0,
                 'source_random_sigma_max': 7.0,
                 'source_random_min_separation': 0.0,
                 'source_random_lrs_spacing': 10.0,
                 'source_random_detection_threshold': 0.2,
                 'source_hotspot_centers': [
                     10.0, 10.0,
                     40.0, 10.0,
                     15.0, 40.0,
                     40.0, 40.0,
                 ],
                 'source_hotspot_weights': [0.4, 0.3, 0.2, 0.1],
                 'source_hotspot_jitter_sigma': 3.0,
                 'persist_source_state': True,
                 'source_state_file':
                     '~/.ros/icir_cleanroom/gas_source_state.json',
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
            launch_arguments={'use_sim_time': 'true',
                              'params_file': empty_nav2_params,
                              'autostart': 'true'}.items())]),
        TimerAction(period=6.0, actions=[Node(
            package='icir_cleanroom', executable='gas_mapping_controller_node.py',
            parameters=[{
                'use_sim_time': True,
                'lrs_dwell_seconds': 1.0,
                'hrs_dwell_seconds': 2.0,
                'max_retries': 2,
                'max_cell_failures': 3,
                'observation_variance_scale': 0.01,
                'observation_variance_floor': 0.001,
                'smoothness_precision': 1.0,
                # Preserve total smoothness while increasing diagonal coupling.
                'cardinal_weight': 0.75, 'diagonal_weight': 0.25,
                'background_precision': 1.0, 'background_mean': 0.0,
                'estimate_resolution': 0.1,
                'log_concentration_min': -4.0,
                'gabp_max_iterations': 500,
                'gabp_tolerance': 1.0e-6,
                'gabp_damping': 0.5,
                'gabp_retry_damping': 0.25,
                'gabp_cg_warning_tolerance': 1.0e-4,
                # UCB peak-search reward and adaptive HRS stopping rule.
                'hrs_ucb_k': 1.0,
                'hrs_stop_margin': 0.02,
                'hrs_min_cycles_per_alert': 1,
                'hrs_max_cycles_per_alert': 10,
                'hrs_peak_improvement_epsilon': 0.001,
                'hrs_peak_max_moves': 10,
                'hrs_candidate_count': 15,
                'hrs_visit_count': 10,
                'hrs_speed': 5.0,
                'hrs_update_seconds': 50.0,
                'hrs_combination_time_limit': 5.0,
                # Use paper_exhaustive_milp only for procedure comparisons.
                'hrs_planner_mode': 'reward_ordered_exact',
                'hazard_threshold': 0.2,
                'history_top_k': 3,
                'history_merge_radius': 5.0,
                'lrs_history_replace_radius': 5.0,
                'lrs_priority_candidate_count': 15,
                'lrs_priority_count': 6,
                'lrs_route_length_ratio': 1.10,
                'lrs_reward_recurrence_weight': 0.40,
                'lrs_reward_severity_weight': 0.20,
                'lrs_reward_staleness_weight': 0.25,
                'lrs_reward_uncertainty_weight': 0.15,
                'history_recent_alpha': 0.5,
                'history_event_half_life': 10.0,
                'history_event_kernel_sigma': 5.0,
                'history_file': '~/.ros/icir_cleanroom/gas_history.json',
            }], output='screen')]),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz], output='screen'),
    ])
