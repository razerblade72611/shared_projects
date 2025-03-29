#!/usr/bin/env python3
"""
Spawns controller spawners for the PLEN robot in the /plen namespace.
This file spawns:
  - The joint_state_broadcaster
  - The joint_trajectory_controller

TimerAction delays the spawners to allow the controller manager to finish initialization.
"""
import os
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    spawn_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/plen/controller_manager'],
        output='screen'
    )

    spawn_joint_traj_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller', '--controller-manager', '/plen/controller_manager'],
        output='screen'
    )

    return LaunchDescription([
        TimerAction(period=3.0, actions=[spawn_joint_state_broadcaster]),
        TimerAction(period=3.5, actions=[spawn_joint_traj_controller])
    ])

if __name__ == '__main__':
    generate_launch_description()

