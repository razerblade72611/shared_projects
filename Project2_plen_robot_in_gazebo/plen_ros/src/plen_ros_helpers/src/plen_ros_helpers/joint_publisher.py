#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import time
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

class JointPub:
    def __init__(self, node: Node):
        self.node = node
        # RIGHT LEG
        self.rhip = self.node.create_publisher(Float64, '/plen/j1_pc/command', 1)
        self.rthigh = self.node.create_publisher(Float64, '/plen/j2_pc/command', 1)
        self.rknee = self.node.create_publisher(Float64, '/plen/j3_pc/command', 1)
        self.rshin = self.node.create_publisher(Float64, '/plen/j4_pc/command', 1)
        self.rankle = self.node.create_publisher(Float64, '/plen/j5_pc/command', 1)
        self.rfoot = self.node.create_publisher(Float64, '/plen/j6_pc/command', 1)
        # LEFT LEG
        self.lhip = self.node.create_publisher(Float64, '/plen/j7_pc/command', 1)
        self.lthigh = self.node.create_publisher(Float64, '/plen/8_pc/command', 1)
        self.lknee = self.node.create_publisher(Float64, '/plen/j9_pc/command', 1)
        self.lshin = self.node.create_publisher(Float64, '/plen/j10_pc/command', 1)
        self.lankle = self.node.create_publisher(Float64, '/plen/j11_pc/command', 1)
        self.lfoot = self.node.create_publisher(Float64, '/plen/j12_pc/command', 1)
        # RIGHT ARM
        self.rshoulder = self.node.create_publisher(Float64, '/plen/j13_pc/command', 1)
        self.rarm = self.node.create_publisher(Float64, '/plen/j14_pc/command', 1)
        self.relbow = self.node.create_publisher(Float64, '/plen/j15_pc/command', 1)
        # LEFT ARM
        self.lshoulder = self.node.create_publisher(Float64, '/plen/j16_pc/command', 1)
        self.larm = self.node.create_publisher(Float64, '/plen/j17_pc/command', 1)
        self.lelbow = self.node.create_publisher(Float64, '/plen/j18_pc/command', 1)

        self.publishers_array = [
            self.rhip, self.rthigh, self.rknee, self.rshin, self.rankle,
            self.rfoot, self.lhip, self.lthigh, self.lknee, self.lshin,
            self.lankle, self.lfoot, self.rshoulder, self.rarm, self.relbow,
            self.lshoulder, self.larm, self.lelbow
        ]
        # Initial joint state: all zeros for 18 joints.
        self.init_pos = np.zeros(18)

    def set_init_pose(self):
        """Sets joints to the initial zero position."""
        self.move_joints(self.init_pos)

    def check_joints_connection(self):
        """
        Loops until each publisher has at least one subscription.
        (Note: In ROS2, get_subscription_count() returns the number of subscriptions.)
        """
        i = 0
        for pub in self.publishers_array:
            while pub.get_subscription_count() == 0:
                i += 1
                self.node.get_logger().debug(f"No subscribers to joint {i} yet, waiting...")
                time.sleep(0.1)
            self.node.get_logger().debug(f"Joint {i} publisher connected")
        self.node.get_logger().debug("All publishers READY")

    def joint_mono_des_callback(self, msg: JointState):
        self.node.get_logger().debug(f"Received joint positions: {msg.position}")
        self.move_joints(msg.position)

    def move_joints(self, joints_array, epsilon=0.05, update_rate=10, time_sleep=0.05, check_position=False):
        """
        Publishes a Float64 command for each joint.
        Optionally waits for confirmation via joint state feedback.
        """
        for i, pub in enumerate(self.publishers_array):
            joint_value = Float64()
            joint_value.data = float(joints_array[i])
            self.node.get_logger().debug(f"Publishing joint {i} value: {joint_value.data}")
            pub.publish(joint_value)
        if check_position:
            self.wait_time_for_execute_movement(joints_array, epsilon, update_rate)
        else:
            self.wait_time_movement_hard(time_sleep)

    def wait_time_for_execute_movement(self, joints_array, epsilon, update_rate):
        """
        Waits until the joint state feedback is within epsilon of the desired values.
        (For simplicity, only the first three joint positions are checked here.)
        """
        self.node.get_logger().debug("START wait_until_jointstate_achieved...")
        start_time = self.node.get_clock().now().nanoseconds / 1e9
        end_time = 0.0
        self.node.get_logger().debug(f"Desired joint state: {joints_array}")
        self.node.get_logger().debug(f"Epsilon: {epsilon}")
        while rclpy.ok():
            current_joint_states = self._check_joint_states_ready()
            values_to_check = current_joint_states.position[:3]
            if self.check_array_similar(joints_array, values_to_check, epsilon):
                self.node.get_logger().debug("Reached joint states!")
                end_time = self.node.get_clock().now().nanoseconds / 1e9
                break
            self.node.get_logger().debug("Not there yet, keep waiting...")
            time.sleep(1.0 / update_rate)
        delta_time = end_time - start_time
        self.node.get_logger().debug(f"Wait time: {delta_time}")
        self.node.get_logger().debug("END wait_until_jointstate_achieved...")
        return delta_time

    def wait_time_movement_hard(self, time_sleep):
        """Performs a hard wait to allow actions to take effect."""
        self.node.get_logger().debug(f"Hard wait: {time_sleep}")
        time.sleep(time_sleep)

    def _check_joint_states_ready(self):
        """
        Helper that subscribes to /plen/joint_states and returns the first received message.
        """
        from sensor_msgs.msg import JointState  # local import
        self.node.get_logger().debug("Waiting for /plen/joint_states to be READY...")
        msg_container = {'msg': None}
        def callback(msg):
            msg_container['msg'] = msg
        sub = self.node.create_subscription(JointState, "/plen/joint_states", callback, 10)
        start_time = self.node.get_clock().now().nanoseconds / 1e9
        timeout = 1.0
        while msg_container['msg'] is None and (self.node.get_clock().now().nanoseconds / 1e9 - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        sub.destroy()
        if msg_container['msg'] is None:
            raise Exception("Timeout waiting for message on /plen/joint_states")
        self.node.get_logger().debug("Joint states received")
        return msg_container['msg']

    def check_array_similar(self, arr1, arr2, epsilon):
        """Returns True if arr1 and arr2 are elementwise equal within epsilon."""
        return np.allclose(arr1, arr2, atol=epsilon)


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('joint_pub_node')
    jp = JointPub(node)
    jp.set_init_pose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

