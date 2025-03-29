#!/usr/bin/env python3
"""
gazebo_plen_launch.py

Launches Gazebo with the default empty world (from /opt/ros/humble/share/gazebo_ros/worlds/empty.world)
using the --verbose flag. It processes the URDF from a Xacro file, starts a robot_state_publisher
(in namespace 'plen'), and spawns the PLEN robot (with entity name "plen_1") into simulation.

NOTE: If an entity with that name already exists, the spawn service will fail.
Be sure to close Gazebo between launches or change the entity name.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, Command, FindExecutable
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition

def generate_launch_description():
    pkg_plen_ros = get_package_share_directory('plen_ros')
    
    # Use the default empty world provided by Gazebo
    default_world = '/opt/ros/humble/share/gazebo_ros/worlds/empty.world'

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    gui          = LaunchConfiguration('gui', default='true')
    headless     = LaunchConfiguration('headless', default='false')
    debug        = LaunchConfiguration('debug', default='false')

    declare_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time')
    declare_gui      = DeclareLaunchArgument('gui', default_value='true', description='Launch Gazebo GUI')
    declare_headless = DeclareLaunchArgument('headless', default_value='false', description='Headless mode')
    declare_debug    = DeclareLaunchArgument('debug', default_value='false', description='Debug mode')

    # Process the Xacro file to generate the robot description.
    xacro_file = os.path.join(pkg_plen_ros, 'urdf', 'plen.urdf.xacro')
    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', xacro_file]),
        value_type=str
    )

    # Launch a robot_state_publisher in the "plen" namespace
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='plen',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': use_sim_time}],
        remappings=[('joint_states', '/plen/joint_states')]
    )

    # Launch gzserver with the default empty world and required ROS plugins.
    gzserver_cmd = ExecuteProcess(
        cmd=[
            'gzserver', '--verbose', default_world,
            '-slibgazebo_ros_init.so', '-slibgazebo_ros_factory.so'
        ],
        output='screen',
        env={
            'GAZEBO_PLUGIN_PATH': '/opt/ros/humble/lib',
            'HOME': os.environ.get('HOME', '/home/razerblade')
        }
    )

    # Launch gzclient only if GUI is enabled.
    gzclient_cmd = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
        condition=IfCondition(gui)
    )

    # Generate URDF from Xacro to a temporary file for spawning.
    xacro_to_urdf = ExecuteProcess(
        cmd=['xacro', xacro_file, '-o', '/tmp/plen.urdf'],
        output='screen'
    )

    # Spawn the robot using the generated URDF.
    spawn_plen = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-entity', 'plen_1',
            '-file', '/tmp/plen.urdf',
            '-x', '0.0', '-y', '0.0', '-z', '0.158'
        ]
    )
    spawn_delay = TimerAction(period=3.0, actions=[spawn_plen])

    return LaunchDescription([
        declare_sim_time,
        declare_gui,
        declare_headless,
        declare_debug,
        rsp_node,
        gzserver_cmd,
        gzclient_cmd,
        xacro_to_urdf,
        spawn_delay
    ])

if __name__ == '__main__':
    generate_launch_description()

