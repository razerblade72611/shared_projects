#!/usr/bin/env python3
"""
gazebo_plen_launch.py

Launches Gazebo with an empty world, loads the URDF from a xacro file,
starts robot_state_publisher, and spawns the robot (named "plen") into simulation.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_plen_ros = get_package_share_directory('plen_ros')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    gui = LaunchConfiguration('gui', default='true')
    headless = LaunchConfiguration('headless', default='false')
    debug = LaunchConfiguration('debug', default='false')

    # Build the robot description by processing the xacro file.
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

    # Start the robot_state_publisher node with the generated robot description.
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time
        }]
    )

    # Launch Gazebo (server and client) using the standard Gazebo launch file.
    empty_world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'debug': debug,
            'gui': gui,
            'paused': 'true',
            'use_sim_time': use_sim_time,
            'headless': headless
        }.items()
    )

    # Spawn the robot into Gazebo from the 'robot_description' topic.
    spawn_urdf = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'plen',
            '-topic', 'robot_description',
            '-x', '0.0', '-y', '0.0', '-z', '0.158'
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('debug', default_value='false'),
        rsp_node,
        empty_world_launch,
        spawn_urdf
    ])

if __name__ == '__main__':
    generate_launch_description()

