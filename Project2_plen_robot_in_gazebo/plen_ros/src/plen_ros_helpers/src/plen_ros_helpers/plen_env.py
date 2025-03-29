#!/usr/bin/env python3
"""
Converted from ROS1 to ROS2 (Humble on Ubuntu 22.04)
File: plen_env.py
Superclass for all PLEN environments.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import time

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Vector3

# Parent Robot Environment for Gym (assumed already ported)
from plen_ros_helpers.robot_gazebo_env import RobotGazeboEnv
# Joint Trajectory Publisher (assumed already ported to ROS2)
from plen_ros_helpers.joint_traj_publisher import JointTrajPub

class PlenEnv(RobotGazeboEnv):
    """Superclass for all PLEN environments."""
    def __init__(self):
        # It is assumed that the parent class creates a node stored in self.node.
        self.node.get_logger().debug("Start PlenEnv INIT...")

        # Namespace and controllers
        self.robot_name_space = "plen"
        self.controllers_list = [
            'rb_servo_r_hip', 'r_hip_r_thigh', 'r_thigh_r_knee',
            'r_knee_r_shin', 'r_shin_r_ankle', 'r_ankle_r_foot',
            'lb_servo_l_hip', 'l_hip_l_thigh', 'l_thigh_l_knee',
            'l_knee_l_shin', 'l_shin_l_ankle', 'l_ankle_l_foot',
            'torso_r_shoulder', 'r_shoulder_rs_servo', 're_servo_r_elbow',
            'torso_l_shoulder', 'l_shoulder_ls_servo', 'le_servo_l_elbow'
        ]
        self.controllers_string = (
            "rb_servo_r_hip r_hip_r_thigh " +
            "r_thigh_r_knee r_knee_r_shin " +
            "r_shin_r_ankle r_ankle_r_foot " +
            "lb_servo_l_hip l_hip_l_thigh " +
            "l_thigh_l_knee l_knee_l_shin " +
            "l_shin_l_ankle l_ankle_l_foot " +
            "torso_r_shoulder " +
            "r_shoulder_rs_servo " +
            "re_servo_r_elbow torso_l_shoulder " +
            "l_shoulder_ls_servo le_servo_l_elbow"
        )
        # Create joint trajectory publisher (assumed to require the node)
        self.joints = JointTrajPub(self.controllers_list, self.controllers_string, self.node)
        self.init_pose = self.joints.jtp_zeros

        # Call parent constructor passing required parameters
        super().__init__(controllers_list=self.controllers_list,
                         robot_name_space=self.robot_name_space,
                         reset_controls=True,
                         start_init_physics_parameters=True,
                         reset_world_or_sim="WORLD")
        
        self.node.get_logger().debug("PlenEnv unpause...")
        # (Optionally, unpause simulation or reset controllers here.)
        self.node.get_logger().debug("Finished PlenEnv INIT...")

    # Helper to wait for a message on a topic
    def _wait_for_message(self, topic, msg_type, timeout=1.0):
        msg_container = {'msg': None}
        def callback(msg):
            msg_container['msg'] = msg
        sub = self.node.create_subscription(msg_type, topic, callback, 10)
        start_time = self.node.get_clock().now().nanoseconds / 1e9
        while msg_container['msg'] is None and (self.node.get_clock().now().nanoseconds / 1e9 - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        sub.destroy()
        if msg_container['msg'] is None:
            self.node.get_logger().error(f"Timeout waiting for message on {topic}")
        return msg_container['msg']

    def _check_all_systems_ready(self):
        """Checks that all sensors and simulation systems are operational."""
        self._check_all_sensors_ready()
        return True

    def _check_all_sensors_ready(self):
        self._check_odom_ready()
        # Uncomment the next line if IMU is required.
        # self._check_imu_ready()
        self._check_rightfoot_contactsensor_state_ready()
        self._check_joint_states_ready()

    def _check_odom_ready(self):
        self.odom = None
        while self.odom is None and rclpy.ok():
            self.odom = self._wait_for_message("/plen/odom", Odometry, timeout=1.0)
        return self.odom

    def _check_imu_ready(self):
        self.imu = None
        while self.imu is None and rclpy.ok():
            self.imu = self._wait_for_message("/plen/imu/data", Imu, timeout=1.0)
        return self.imu

    def _check_rightfoot_contactsensor_state_ready(self):
        self.rightfoot_contactsensor_state = None
        while self.rightfoot_contactsensor_state is None and rclpy.ok():
            self.rightfoot_contactsensor_state = self._wait_for_message("/plen/right_foot_contact", ContactsState, timeout=1.0)
        return self.rightfoot_contactsensor_state

    def _check_leftfoot_contactsensor_state_ready(self):
        self.leftfoot_contactsensor_state = None
        while self.leftfoot_contactsensor_state is None and rclpy.ok():
            self.leftfoot_contactsensor_state = self._wait_for_message("/plen/left_foot_contact", ContactsState, timeout=1.0)
        return self.leftfoot_contactsensor_state

    def _check_joint_states_ready(self):
        self.joint_states = None
        while self.joint_states is None and rclpy.ok():
            self.joint_states = self._wait_for_message("/plen/joint_states", JointState, timeout=1.0)
        return self.joint_states

