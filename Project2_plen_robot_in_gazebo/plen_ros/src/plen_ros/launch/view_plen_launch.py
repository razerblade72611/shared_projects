#!/usr/bin/env python3
"""
view_plen_launch.py

Launches a view-only configuration: loads the robot description,
starts robot_state_publisher, launches the joint_state_publisher_gui
(if requested) and opens RViz with a preconfigured view.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    pkg_plen_ros = get_package_share_directory('plen_ros')
    urdf_file = os.path.join(pkg_plen_ros, 'urdf', 'plen.urdf.xacro')
    rviz_config_file = os.path.join(pkg_plen_ros, 'rviz', 'plen.rviz')

    # Launch argument to optionally launch the joint_state_publisher GUI.
    use_jsp_gui = LaunchConfiguration('use_jsp_gui', default='True')

    # Build the robot_description by processing the xacro.
    robot_description = Command(['xacro ', urdf_file])

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    jsp_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        condition=IfCondition(use_jsp_gui)
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_jsp_gui', default_value='True',
                              description='Launch joint_state_publisher_gui'),
        rsp_node,
        jsp_gui_node,
        rviz_node
    ])

