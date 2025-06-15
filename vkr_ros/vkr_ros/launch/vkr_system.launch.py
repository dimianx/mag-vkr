import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('vkr_ros')
    config_dir = os.path.join(pkg_dir, 'config')
    params_file = os.path.join(config_dir, 'vkr_params.yaml')
    
    num_uavs = LaunchConfiguration('num_uavs', default='3')
    terrain_file = LaunchConfiguration('terrain_file')
    obstacle_files = LaunchConfiguration('obstacle_files')
    obstacle_transforms = LaunchConfiguration('obstacle_transforms')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'num_uavs',
            default_value='3',
            description='Number of UAVs in the system'
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
            description='List of obstacle transforms (9 values per obstacle: x,y,z,roll,pitch,yaw,sx,sy,sz)'
        ),
        
        Node(
            package='vkr_ros',
            executable='collision_server',
            name='collision_server',
            output='screen',
            parameters=[
                params_file,
                {
                    'terrain_file': terrain_file,
                    'obstacle_files': PythonExpression([obstacle_files]),
                    'obstacle_transforms': PythonExpression([obstacle_transforms])
                }
            ]
        ),
        
        Node(
            package='vkr_ros',
            executable='corridor_manager',
            name='corridor_manager',
            output='screen',
            parameters=[
                params_file,
                {
                    'max_uavs': PythonExpression(["int(", num_uavs, ")"])
                }
            ]
        ),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(pkg_dir, 'launch', 'uav_nodes.launch.py')
            ]),
            launch_arguments={
                'num_uavs': num_uavs
            }.items()
        ),
    ])