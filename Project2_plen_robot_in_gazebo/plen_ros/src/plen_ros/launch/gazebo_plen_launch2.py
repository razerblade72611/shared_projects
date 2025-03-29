#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
from launch_ros.parameter_descriptions import ParameterValue # ADD THIS IMPORT

def generate_launch_description():
    # Declare launch arguments (keep these as they might be used later)
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true', description='Use simulation time')
    gui_arg = DeclareLaunchArgument(
        'gui', default_value='true', description='Enable GUI')
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false', description='Run headless')
    debug_arg = DeclareLaunchArgument(
        'debug', default_value='false', description='Enable debug mode')

    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')
    headless = LaunchConfiguration('headless')
    debug = LaunchConfiguration('debug')

    # Compute the path to the xacro file and process it (as before)
    pkg_plen_ros = get_package_share_directory('plen_ros')
    xacro_file = os.path.join(pkg_plen_ros, 'urdf', 'plen.urdf.xacro')
    robot_description_content = Command(['xacro ', xacro_file])

    # **Wrap robot_description_content in ParameterValue with value_type=str**
    robot_description = {'robot_description': ParameterValue(value=robot_description_content, value_type=str)} # MODIFIED

    # Launch gzserver (Gazebo server) - as before
    start_gazebo_server_cmd = [
        'gzserver',
        '--verbose',
        '-s', 'libgazebo_ros_init.so',
        '-s', 'libgazebo_ros_factory.so',
    ]
    if debug:
        start_gazebo_server_cmd.append('--pause')

    start_gazebo_server = ExecuteProcess(
        cmd=start_gazebo_server_cmd,
        output='screen'
    )

    # Launch gzclient (Gazebo client/GUI) - as before
    start_gazebo_client = ExecuteProcess(
        condition=IfCondition(gui),
        cmd=['gzclient'],
        output='screen'
    )

    # Spawn the URDF model in Gazebo using spawn_entity.py (ROS 2 equivalent of spawn_model) - MODIFIED parameters
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_urdf',
        output='screen',
        arguments=[
            '-entity', 'plen',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.158'
        ],
        parameters=[robot_description] # MODIFIED - Pass the dict with ParameterValue
    )

    return LaunchDescription([
        use_sim_time_arg,
        gui_arg,
        headless_arg,
        debug_arg,
        start_gazebo_server,
        start_gazebo_client,
        spawn_entity
    ])
