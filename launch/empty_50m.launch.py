import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_dir = get_package_share_directory('icir_cleanroom')
    return LaunchDescription([
        DeclareLaunchArgument('save_history', default_value='true',
                              choices=['true', 'false'],
                              description='Write history files; existing history is still loaded'),
        DeclareLaunchArgument(
            'repeat_after_hrs', default_value='true', choices=['true', 'false'],
            description='Repeat after HRS; false stops mapping after the first HRS search'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                package_dir, 'launch', 'gas_mapping.launch.py')),
            launch_arguments={
                'save_history': LaunchConfiguration('save_history'),
                'repeat_after_hrs': LaunchConfiguration('repeat_after_hrs'),
                'environment': 'empty_50m',
            }.items())])
