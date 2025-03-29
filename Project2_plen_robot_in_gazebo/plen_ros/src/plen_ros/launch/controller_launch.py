#!/usr/bin/env python3
"""
controller_launch.py

Launches individual joint position controllers (j1_pc ... j18_pc)
and the joint state broadcaster (remapped to /plen/joint_states).
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.substitutions import FindPackageShare  # (not used, but can be removed if unused)
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_share = get_package_share_directory('plen_ros')
    urdf_file = os.path.join(pkg_share, 'urdf', 'plen.urdf.xacro')

    # Ensure robot_description is provided as string ParameterValue
    robot_description = ParameterValue(Command(['xacro', urdf_file]), value_type=str)

    # Robot State Publisher (using robot_description, remap joint_states to /plen namespace)
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
        remappings=[('/joint_states', '/plen/joint_states')]
    )

    # (Optional) Launch controller manager with parameter file, if not using Gazebo plugin.
    # cm_node = Node(
    #     package='controller_manager',
    #     executable='ros2_control_node',
    #     parameters=[os.path.join(pkg_share, 'config', 'plen_position_controllers.yaml')],
    #     output='screen'
    # )

    # Spawn all individual position controllers for joints j1...j18 and the joint state broadcaster
    controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace='/plen',
        output='screen',
        arguments=[
            'j1_pc', 'j2_pc', 'j3_pc', 'j4_pc', 'j5_pc', 'j6_pc',
            'j7_pc', 'j8_pc', 'j9_pc', 'j10_pc', 'j11_pc', 'j12_pc',
            'j13_pc', 'j14_pc', 'j15_pc', 'j16_pc', 'j17_pc', 'j18_pc',
            'joint_state_broadcaster'  # use the broadcaster to publish joint states
        ]
    )

    return LaunchDescription([
        rsp_node,
        controller_spawner
    ])

