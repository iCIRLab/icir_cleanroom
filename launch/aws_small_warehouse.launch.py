"""Compatibility entry point for the AWS Small Warehouse environment."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    common_launch = os.path.join(
        get_package_share_directory('icir_cleanroom'),
        'launch', 'gas_mapping.launch.py')
    return LaunchDescription([
        DeclareLaunchArgument(
            'repeat_after_hrs', default_value='true', choices=['true', 'false'],
            description='Repeat after HRS; false stops mapping after the first HRS search'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(common_launch),
            launch_arguments={
                'repeat_after_hrs': LaunchConfiguration('repeat_after_hrs'),
                'environment': 'aws_small_warehouse',
            }.items())])
