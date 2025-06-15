import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    px4_dir = os.environ.get('PX4_DIR', os.path.expanduser('~/PX4-Autopilot'))
    
    num_uavs = LaunchConfiguration('num_uavs', default='3')
    world = LaunchConfiguration('world')
    
    ld = LaunchDescription([
        DeclareLaunchArgument(
            'num_uavs',
            default_value='3',
            description='Number of UAVs to spawn'
        ),
        
        DeclareLaunchArgument(
            'world',
            description='Path to Gazebo world file'
        ),
        
        ExecuteProcess(
            cmd=[
                'gazebo',
                '--verbose',
                '-s', 'libgazebo_ros_init.so',
                '-s', 'libgazebo_ros_factory.so',
                world
            ],
            output='screen',
            name='gazebo'
        ),
    ])
    
    num_vehicles = int(LaunchConfiguration('num_uavs', default='3').perform(None))
    
    for i in range(num_vehicles):
        instance = i
        
        px4_instance = ExecuteProcess(
            cmd=[
                px4_dir + '/build/px4_sitl_default/bin/px4',
                px4_dir + '/ROMFS/px4fmu_common',
                '-s', 'etc/init.d-posix/rcS',
                '-i', str(instance),
                '-d'
            ],
            cwd=px4_dir,
            output='screen',
            name=f'px4_instance_{instance}',
            env={
                **os.environ,
                'PX4_SIM_MODEL': 'iris',
                'PX4_HOME_LAT': '47.641468',
                'PX4_HOME_LON': '-122.140165',
                'PX4_HOME_ALT': '0.0',
                'PX4_SIM_HOST_ADDR': '127.0.0.1',
                'PX4_INSTANCE': str(instance),
                'PX4_SIM_PORT_OFFSET': str(instance * 10),
            }
        )
        
        spawn_model = ExecuteProcess(
            cmd=[
                'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                '-entity', f'iris_{instance}',
                '-database', 'iris',
                '-x', str(i * 3),
                '-y', '0',
                '-z', '0.1',
                '-R', '0',
                '-P', '0',
                '-Y', '0'
            ],
            output='screen',
            name=f'spawn_iris_{instance}'
        )
        
        ld.add_action(px4_instance)
        ld.add_action(spawn_model)
    
    return ld