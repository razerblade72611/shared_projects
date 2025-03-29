#!/usr/bin/env python3
"""
plen_walk_launch.py

Includes the Gazebo simulation (via gazebo_plen_launch.py), launches the gazebo_tools_test node,
starts the controller manager (ros2_control_node) with the robot description,
includes the trajectory controller spawner launch, and conditionally launches the plen_td3 node.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_plen_ros = get_package_share_directory('plen_ros')

    # Declare launch arguments.
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    gui = LaunchConfiguration('gui', default='true')
    headless = LaunchConfiguration('headless', default='false')
    debug = LaunchConfiguration('debug', default='false')
    td3 = LaunchConfiguration('td3', default='False')

    # Build the robot description from the xacro file.
    robot_description = ParameterValue(
        Command([
            'xacro ',
            PathJoinSubstitution([
                FindPackageShare('plen_ros'),
                'urdf',
                'plen.urdf.xacro'
            ])
        ]),
        value_type=str
    )

    # Launch the controller manager node (ros2_control_node) with the robot description.
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='controller_manager',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time
        }]
    )

    # Include the Gazebo simulation launch.
    gazebo_plen_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_plen_ros, 'launch', 'gazebo_plen_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'gui': gui,
            'headless': headless,
            'debug': debug
        }.items()
    )

    # Launch the gazebo_tools_test node.
    gazebo_tools_test_node = Node(
        package='plen_ros',
        executable='gazebo_tools_test',
        name='gazebo_tools_test',
        output='screen'
    )

    # Include the trajectory controller launch file.
    traj_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_plen_ros, 'launch', 'traj_controller_launch.py')
        )
    )

    # Conditionally launch the plen_td3 node if the "td3" argument is True.
    plen_td3_node = Node(
        package='plen_ros',
        executable='plen_td3',
        name='plen_td3',
        output='screen',
        condition=IfCondition(td3)
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('debug', default_value='false'),
        DeclareLaunchArgument('td3', default_value='False'),
        controller_manager_node,
        gazebo_plen_launch,
        gazebo_tools_test_node,
        traj_controller_launch,
        plen_td3_node
    ])

if __name__ == '__main__':
    generate_launch_description()

