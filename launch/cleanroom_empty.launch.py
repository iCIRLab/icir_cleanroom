"""Backward-compatible entry point for the empty 50 m environment."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_dir = get_package_share_directory('icir_cleanroom')
    return LaunchDescription([
        DeclareLaunchArgument(
            'hrs_candidate_threshold',
            description='Required HRS posterior-score candidate threshold'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                package_dir, 'launch', 'gas_mapping.launch.py')),
            launch_arguments={
                'environment': 'empty_50m',
                'hrs_candidate_threshold': LaunchConfiguration(
                    'hrs_candidate_threshold'),
            }.items())])
