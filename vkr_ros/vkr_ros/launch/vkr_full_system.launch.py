import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_dir = get_package_share_directory('vkr_ros')
    
    num_uavs = LaunchConfiguration('num_uavs', default='3')
    world = LaunchConfiguration('world', default='empty.world')
    terrain_file = LaunchConfiguration('terrain_file', default='')
    obstacle_files = LaunchConfiguration('obstacle_files', default='[]')
    obstacle_transforms = LaunchConfiguration('obstacle_transforms', default='[]')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'num_uavs',
            default_value='3',
            description='Number of UAVs in the system'
        ),
        
        DeclareLaunchArgument(
            'world',
            description='Path to Gazebo world file'
        ),
        
        DeclareLaunchArgument(
            'terrain_file',
            default_value='',
            description='Path to terrain file'
        ),
        
        DeclareLaunchArgument(
            'obstacle_files',
            default_value='[]',
            description='List of obstacle OBJ files'
        ),
        
        DeclareLaunchArgument(
            'obstacle_transforms',
            default_value='[]',
            description='List of obstacle transforms'
        ),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(pkg_dir, 'launch', 'px4_multi_sitl.launch.py')
            ]),
            launch_arguments={
                'num_uavs': num_uavs,
                'world': world
            }.items()
        ),
        
        TimerAction(
            period=10.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        os.path.join(pkg_dir, 'launch', 'mavros_multi.launch.py')
                    ]),
                    launch_arguments={
                        'num_uavs': num_uavs
                    }.items()
                ),
            ]
        ),
        
        TimerAction(
            period=15.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        os.path.join(pkg_dir, 'launch', 'vkr_system.launch.py')
                    ]),
                    launch_arguments={
                        'num_uavs': num_uavs,
                        'terrain_file': terrain_file,
                        'obstacle_files': obstacle_files,
                        'obstacle_transforms': obstacle_transforms
                    }.items()
                ),
            ]
        ),
    ])