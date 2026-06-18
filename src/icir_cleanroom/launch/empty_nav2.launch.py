import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    pkg_dir = get_package_share_directory('icir_cleanroom')

    nav2_params_path = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    map_path = os.path.join(pkg_dir, 'map', 'empty.yaml')

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
            'params_file': nav2_params_path
        }.items()
    )

    return LaunchDescription([
        nav2_bringup_launch,
    ])