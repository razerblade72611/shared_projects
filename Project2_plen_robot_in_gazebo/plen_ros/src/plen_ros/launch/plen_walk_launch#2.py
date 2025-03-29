#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution, FindExecutable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_plen_ros = get_package_share_directory('plen_ros')

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')
    headless = LaunchConfiguration('headless')
    debug = LaunchConfiguration('debug')
    td3 = LaunchConfiguration('td3')

    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')
    gui_arg = DeclareLaunchArgument('gui', default_value='true')
    headless_arg = DeclareLaunchArgument('headless', default_value='false')
    debug_arg = DeclareLaunchArgument('debug', default_value='false')
    td3_arg = DeclareLaunchArgument('td3', default_value='False')

    # ✅ Fix: Use FindExecutable and proper Command formatting
    xacro_file = os.path.join(pkg_plen_ros, 'urdf', 'plen.urdf.xacro')
    robot_description_content = Command([FindExecutable(name='xacro'), xacro_file])
    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}

    # Include Gazebo launch
    gazebo_plen_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_plen_ros, 'launch', 'gazebo_plen_launch.py')
        ),
        launch_arguments={
            'debug': debug,
            'gui': gui,
            'use_sim_time': use_sim_time,
            'headless': headless
        }.items()
    )

    # Include trajectory controller launch
    traj_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_plen_ros, 'launch', 'traj_controller_launch.py')
        )
    )

    # Gazebo tools test node
    gazebo_tools_test_node = Node(
        package='plen_ros',
        executable='gazebo_tools_test',
        name='gazebo_tools_test',
        output='screen'
    )

    # Optional TD3 node
    plen_td3_node = Node(
        package='plen_ros',
        executable='plen_td3',
        name='plen_td3',
        output='screen',
        condition=IfCondition(td3)
    )

    # ✅ Fix: Properly set Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        use_sim_time_arg,
        gui_arg,
        headless_arg,
        debug_arg,
        td3_arg,
        gazebo_plen_launch,
        traj_controller_launch,
        gazebo_tools_test_node,
        plen_td3_node,
        robot_state_publisher
    ])

