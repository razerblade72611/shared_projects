#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.srv import JointRequest  # Assumed to be available in ROS2

class JointTrajPub:
    def __init__(self, node: Node, joint_name_list, joint_name_string):
        self.node = node
        self.jtp = self.node.create_publisher(JointTrajectory, '/plen/joint_trajectory_controller/command', 1)
        self.clear_forces_client = self.node.create_client(JointRequest, "/gazebo/clear_joint_forces")
        self.joint_name_list = joint_name_list
        # Convert zeros array to a Python list
        self.jtp_zeros = np.zeros(len(joint_name_list)).tolist()
        self.joint_name_string = joint_name_string

    def move_joints(self, pos):
        self.check_joints_connection()
        if not self.clear_forces_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error("Service /gazebo/clear_joint_forces not available!")
            return
        for name in self.joint_name_list:
            req = JointRequest.Request()
            req.joint_name = name
            future = self.clear_forces_client.call_async(req)
            rclpy.spin_until_future_complete(self.node, future)
            if future.result() is None:
                self.node.get_logger().error(f"Failed to clear forces for joint {name}")
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_name_list
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jtp_zeros
        point.accelerations = self.jtp_zeros
        point.effort = (np.ones(len(self.joint_name_list)) * 0.15).tolist()
        # Set the minimal time_from_start: 1 nanosecond
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 1
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)

    def set_init_pose(self, pos):
        self.check_joints_connection()
        if not self.clear_forces_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error("Service /gazebo/clear_joint_forces not available!")
            return
        for name in self.joint_name_list:
            req = JointRequest.Request()
            req.joint_name = name
            future = self.clear_forces_client.call_async(req)
            rclpy.spin_until_future_complete(self.node, future)
        # Publish an empty JointTrajectory to clear previous commands
        empty_msg = JointTrajectory()
        self.jtp.publish(empty_msg)
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_name_list
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jtp_zeros
        point.accelerations = self.jtp_zeros
        point.effort = (np.ones(len(self.joint_name_list)) * 0.15).tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 1
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)

    def check_joints_connection(self):
        """Waits until the joint trajectory publisher has at least one subscriber."""
        while self.jtp.get_subscription_count() == 0:
            self.node.get_logger().debug("No subscribers to joint trajectory publisher yet, waiting...")
            time.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('joint_traj_pub_node')
    # Define your joint names as in your URDF
    joint_name_list = [
        'rb_servo_r_hip', 'r_hip_r_thigh', 'r_thigh_r_knee',
        'r_knee_r_shin', 'r_shin_r_ankle', 'r_ankle_r_foot',
        'lb_servo_l_hip', 'l_hip_l_thigh', 'l_thigh_l_knee',
        'l_knee_l_shin', 'l_shin_l_ankle', 'l_ankle_l_foot',
        'torso_r_shoulder', 'r_shoulder_rs_servo', 're_servo_r_elbow',
        'torso_l_shoulder', 'l_shoulder_ls_servo', 'le_servo_l_elbow'
    ]
    joint_name_string = ""  # Use as needed
    jtp = JointTrajPub(node, joint_name_list, joint_name_string)
    # Set initial pose to all zeros
    init_pose = [0.0] * len(joint_name_list)
    jtp.set_init_pose(init_pose)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

