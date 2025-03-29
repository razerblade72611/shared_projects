#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import time


class AllJoints:
    def __init__(self, node: Node, joint_name_lst):
        self.node = node
        self.node.get_logger().info('Creating joint trajectory Publisher')
        # Create a publisher on the topic /plen/joint_trajectory_controller/command
        self.jtp = self.node.create_publisher(JointTrajectory, '/plen/joint_trajectory_controller/command', 1)
        self.joint_name_lst = joint_name_lst
        # Create a list of zeros with length equal to the number of joints
        self.jtp_zeros = [0.0] * len(joint_name_lst)

    def move_jtp(self, pos):
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_name_lst

        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jtp_zeros
        point.accelerations = self.jtp_zeros
        point.effort = self.jtp_zeros
        # Set time_from_start to 0.0001 seconds
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(0.0001 * 1e9)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)
        self.node.get_logger().info("Published joint trajectory command")

    def reset_move_jtp(self, pos):
        # Publish an empty message first to clear previous commands
        empty_msg = JointTrajectory()
        self.jtp.publish(empty_msg)
        # Now publish the new command
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_name_lst

        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jtp_zeros
        point.accelerations = self.jtp_zeros
        point.effort = self.jtp_zeros
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(0.0001 * 1e9)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)
        self.node.get_logger().info("Reset joint trajectory command published")


class PlenEnvironment:
    def __init__(self, node: Node):
        self.node = node
        self.link_name_lst = [
            'plen::base_footprint', 'plen::l_shoulder', 'plen::ls_servo',
            'plen::l_elbow', 'plen::l_hip', 'plen::l_thigh', 'plen::l_knee',
            'plen::l_shin', 'plen::l_ankle', 'plen::l_foot',
            'plen::r_shoulder', 'plen::rs_servo', 'plen::r_elbow',
            'plen::r_hip', 'plen::r_thigh', 'plen::r_knee', 'plen::r_shin',
            'plen::r_ankle', 'plen::r_foot'
        ]
        self.joint_name_lst = [
            'rb_servo_r_hip', 'r_hip_r_thigh', 'r_thigh_r_knee',
            'r_knee_r_shin', 'r_shin_r_ankle', 'r_ankle_r_foot',
            'lb_servo_l_hip', 'l_hip_l_thigh', 'l_thigh_l_knee',
            'l_knee_l_shin', 'l_shin_l_ankle', 'l_ankle_l_foot',
            'torso_r_shoulder', 'r_shoulder_rs_servo', 're_servo_r_elbow',
            'torso_l_shoulder', 'l_shoulder_ls_servo', 'le_servo_l_elbow'
        ]
        self.all_joints = AllJoints(self.node, self.joint_name_lst)
        self.starting_pos = [0.0] * len(self.joint_name_lst)
        self.joint_pos = self.starting_pos

    def reset(self):
        self.joint_pos = self.starting_pos
        print('RESET:', self.joint_pos)
        self.all_joints.reset_move_jtp(self.starting_pos)

    def step(self, action):
        print('STEP:', action)
        self.joint_pos = action
        self.all_joints.move_jtp(self.joint_pos)


def main():
    rclpy.init()
    node = rclpy.create_node('joint_position_node')
    env = PlenEnvironment(node)

    env.reset()

    for i in range(10):
        if i % 2 == 0:
            print("STEPPING")
            joint_val = [0.0] * 18
            # Set joint index 13 to 1.0 (indexing corresponds to your joint list)
            joint_val[13] = 1.0
            env.step(joint_val)
        else:
            print("RESETTING")
            env.reset()
        time.sleep(1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

