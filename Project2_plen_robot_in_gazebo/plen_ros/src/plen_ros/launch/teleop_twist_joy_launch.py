#!/usr/bin/env python3
"""
teleop_twist_joy_launch.py

Launches the joy_node and teleop_twist_joy node for manual teleoperation.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments.
    joy_dev_arg = DeclareLaunchArgument(
        'joy_dev', default_value='/dev/input/js0',
        description='Device file for the joystick'
    )
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(
            os.getenv('HOME'),
            'plen_ml_walk-master', 'plen_ros', 'config', 'teleop_twist_joy.yaml'
        ),
        description='Path to teleop_twist_joy config file'
    )

    # Launch the joy node that reads joystick input.
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{'dev': LaunchConfiguration('joy_dev')}]
    )

    # Launch the teleop_twist_joy node that converts joystick messages to Twist.
    teleop_twist_joy_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
        # Remap the output Twist topic to your robot’s namespace.
        remappings=[('/cmd_vel', '/plen/cmd_vel')]
    )

    return LaunchDescription([
        joy_dev_arg,
        config_file_arg,
        joy_node,
        teleop_twist_joy_node
    ])

