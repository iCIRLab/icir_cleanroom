import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    pkg_dir = get_package_share_directory('icir_cleanroom')

    nav2_params_path = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')

    map_data_yaml_path = os.path.join(pkg_dir, 'config', 'gas_attraction', 'map_data.yaml')
    idw_yaml_path = os.path.join(pkg_dir, 'config', 'gas_attraction', 'idw.yaml')

    map_yaml_arg = DeclareLaunchArgument(
        'map',
        default_value='empty.yaml',
        description='map yaml file name under icir_cleanroom/map'
    )
    map_path = PathJoinSubstitution([pkg_dir, 'map', LaunchConfiguration('map')])

    configured_params = RewrittenYaml(
        source_file=nav2_params_path,
        root_key='',
        param_rewrites={
            'map_data_yaml': map_data_yaml_path,
            'idw_yaml': idw_yaml_path,
        },
        convert_types=True
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
            'map': map_path,
            'params_file': configured_params
        }.items()
    )

    return LaunchDescription([
        map_yaml_arg,
        nav2_bringup_launch,
    ])