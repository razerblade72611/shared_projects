#!/usr/bin/env python3
"""
Main launch file for PLEN simulation.

This launch file:
  - Includes the Gazebo launch (which processes the URDF and spawns the robot)
  - Launches the controller manager node using the trajectory controller YAML configuration
  - Includes the controller spawner launch (which loads controllers after a delay)
  - Optionally launches extra nodes (e.g., gazebo_tools_test, plen_td3)
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import FindExecutable

def generate_launch_description():
    pkg_plen_ros = get_package_share_directory('plen_ros')

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui          = LaunchConfiguration('gui')
    headless     = LaunchConfiguration('headless')
    debug        = LaunchConfiguration('debug')
    td3          = LaunchConfiguration('td3')
    control_mode = LaunchConfiguration('control_mode', default='trajectory')
    tools        = LaunchConfiguration('gazebo_tools', default='false')

    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation clock')
    gui_arg          = DeclareLaunchArgument('gui', default_value='true', description='Launch Gazebo GUI')
    headless_arg     = DeclareLaunchArgument('headless', default_value='false', description='Headless mode')
    debug_arg        = DeclareLaunchArgument('debug', default_value='false', description='Debug mode')
    td3_arg          = DeclareLaunchArgument('td3', default_value='false', description='Launch TD3 control node')
    control_mode_arg = DeclareLaunchArgument('control_mode', default_value='trajectory', description="Control mode: 'trajectory' or 'position'")
    tools_arg        = DeclareLaunchArgument('gazebo_tools', default_value='false', description='Launch gazebo_tools_test node')

    # Include the Gazebo launch file.
    gazebo_plen_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_plen_ros, 'launch', 'gazebo_plen_launch.py')),
        launch_arguments={
            'debug': debug,
            'gui': gui,
            'use_sim_time': use_sim_time,
            'headless': headless
        }.items()
    )

    # Process the Xacro file for robot_description (for controller manager subscription)
    xacro_file = os.path.join(pkg_plen_ros, 'urdf', 'plen.urdf.xacro')
    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', xacro_file]),
        value_type=str
    )

    # Path to the trajectory controller YAML configuration
    traj_controller_cfg = os.path.join(pkg_plen_ros, 'config', 'plen_trajectory_control.yaml')

    # Launch the controller manager node in the '/plen' namespace.
    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='controller_manager',
        namespace='plen',
        output='screen',
        parameters=[traj_controller_cfg]
    )

    # Include the controller spawner launch file.
    traj_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_plen_ros, 'launch', 'traj_controller_launch.py'))
    )

    # Optional additional nodes
    gazebo_tools_test_node = Node(
        package='plen_ros',
        executable='gazebo_tools_test',
        name='gazebo_tools_test',
        output='screen',
        condition=IfCondition(tools)
    )
    plen_td3_node = Node(
        package='plen_ros',
        executable='plen_td3',
        name='plen_td3',
        output='screen',
        condition=IfCondition(td3)
    )

    return LaunchDescription([
        use_sim_time_arg,
        gui_arg,
        headless_arg,
        debug_arg,
        td3_arg,
        control_mode_arg,
        tools_arg,
        gazebo_plen_launch,
        control_node,
        traj_controller_launch,
        gazebo_tools_test_node,
        plen_td3_node
    ])

if __name__ == '__main__':
    generate_launch_description()

