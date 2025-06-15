import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    pkg_dir = get_package_share_directory('vkr_ros')
    config_dir = os.path.join(pkg_dir, 'config')
    params_file = os.path.join(config_dir, 'vkr_params.yaml')
    
    num_uavs = int(LaunchConfiguration('num_uavs', default='3').perform(None))
    
    ld = LaunchDescription()
    
    for i in range(1, num_uavs + 1):
        uav_namespace = f'uav_{i}'
        mission_file = os.path.join(config_dir, f'mission_{i}.yaml')
        
        uav_group = GroupAction([
            PushRosNamespace(uav_namespace),
            
            Node(
                package='vkr_ros',
                executable='state_machine',
                name='state_machine',
                output='screen',
                parameters=[
                    params_file,
                    {
                        'uav_id': i,
                        'mission_file': mission_file
                    }
                ]
            ),
            
            Node(
                package='vkr_ros',
                executable='planner_node',
                name='planner_node',
                output='screen',
                parameters=[
                    params_file,
                    {
                        'uav_id': i
                    }
                ]
            ),
            
            Node(
                package='vkr_ros',
                executable='landing_detector',
                name='landing_detector',
                output='screen',
                parameters=[params_file]
            ),
        ])
        
        ld.add_action(uav_group)
    
    return ld