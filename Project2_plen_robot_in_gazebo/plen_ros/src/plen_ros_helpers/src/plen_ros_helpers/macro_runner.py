#!/usr/bin/env python3
import rclpy
import json
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class MacroRunner(Node):
    def __init__(self):
        super().__init__('macro_runner')
        
        # Publisher for JointTrajectory messages
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/plen/joint_trajectory_controller/joint_trajectory_command',
            10
        )

        # Load macro file (replace with actual file path if needed)
        with open('/home/razerblade/plen_ml_walk-master/plen_ros/config/plen_walk_forward.json', 'r') as f:
            self.macro = json.load(f)

        self.joint_mapping = {
            "left_shoulder_pitch": "torso_l_shoulder",
            "left_shoulder_roll": "l_shoulder_ls_servo",
            "left_elbow_roll": "le_servo_l_elbow",
            "left_thigh_yaw": "lb_servo_l_hip",
            "left_thigh_roll": "l_hip_l_thigh",
            "left_thigh_pitch": "l_thigh_l_knee",
            "left_knee_pitch": "l_knee_l_shin",
            "left_foot_pitch": "l_shin_l_ankle",
            "left_foot_roll": "l_ankle_l_foot",
            "right_shoulder_pitch": "torso_r_shoulder",
            "right_shoulder_roll": "r_shoulder_rs_servo",
            "right_elbow_roll": "re_servo_r_elbow",
            "right_thigh_yaw": "rb_servo_r_hip",
            "right_thigh_roll": "r_hip_r_thigh",
            "right_thigh_pitch": "r_thigh_r_knee",
            "right_knee_pitch": "r_knee_r_shin",
            "right_foot_pitch": "r_shin_r_ankle",
            "right_foot_roll": "r_ankle_r_foot"
        }

        self.execute_macro()

    def execute_macro(self):
        """Convert the macro to a JointTrajectory and publish it."""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = list(self.joint_mapping.values())

        for frame in self.macro["frames"]:
            point = JointTrajectoryPoint()
            positions = [0.0] * len(self.joint_mapping)
            
            # Read joint values from frame
            for output in frame["outputs"]:
                device_name = output["device"]
                if device_name in self.joint_mapping:
                    joint_name = self.joint_mapping[device_name]
                    index = traj_msg.joint_names.index(joint_name)
                    positions[index] = output["value"] * 0.01745  # Convert degrees to radians

            point.positions = positions
            point.time_from_start.sec = frame["transition_time_ms"] // 1000
            point.time_from_start.nanosec = (frame["transition_time_ms"] % 1000) * 1_000_000

            traj_msg.points.append(point)

        # Publish the trajectory
        self.publisher.publish(traj_msg)
        self.get_logger().info("Published macro trajectory.")

def main(args=None):
    rclpy.init(args=args)
    node = MacroRunner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

