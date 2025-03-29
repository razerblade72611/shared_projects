#!/usr/bin/env python3
"""
Converted from ROS1 to ROS2 (Humble on Ubuntu 22.04)
File: plen_walk.py
Defines the Gym environment for PLEN walking.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import time

from gym import spaces
from gym.envs.registration import register

# Import the base PLEN environment (converted to ROS2)
from plen_ros_helpers.plen_env import PlenEnv

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import JointState, Imu
from tf_transformations import euler_from_quaternion  # Ensure tf_transformations is installed
from plen_ros.srv import Iterate  # Assumed converted to ROS2
from std_msgs.msg import Int32

register(
    id='PlenWalkEnv-v0',
    entry_point='plen_ros_helpers.plen_walk:PlenWalkEnv',
    max_episode_steps=500,  # Time step limit per episode
)

class PlenWalkEnv(PlenEnv):
    def __init__(self):
        self.node.get_logger().debug("Start PlenWalkEnv INIT...")
        self.max_episode_steps = 500
        self.init_pose = np.zeros(18)

        # Simulation step timing parameters
        self.running_step = 165e5   # in nanoseconds
        self.running_step_sec = 0.0165

        # Define Agent Action Space
        low_act = np.ones(18) * -1
        high_act = np.ones(18)
        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)

        # Define Environment Action Space ranges
        self.env_ranges = [
            [-1.57, 1.57],  # RIGHT LEG
            [-0.15, 1.5],
            [-0.95, 0.75],
            [-0.9, 0.3],
            [-0.95, 1.2],
            [-0.8, 0.4],
            [-1.57, 1.57],  # LEFT LEG
            [-1.5, 0.15],
            [-0.75, 0.95],
            [-0.3, 0.9],
            [-1.2, 0.95],
            [-0.4, 0.8],
            [-1.57, 1.57],  # RIGHT ARM
            [-0.15, 1.57],
            [-0.2, 0.35],
            [-1.57, 1.57],  # LEFT ARM
            [-1.57, 0.15],
            [-0.35, 0.2]
        ]

        # Reward parameters
        self.reward_range = (-np.inf, np.inf)
        self.dead_penalty = 100.
        self.alive_reward = self.dead_penalty / self.max_episode_steps
        self.vel_weight = 3.
        self.init_height = 0.158
        self.height_weight = 20.
        self.straight_weight = 1
        self.roll_weight = 1.
        self.pitch_weight = 0.5
        self.yaw_weight = 1.
        self.joint_effort_weight = 0.035
        self.dead = False

        # Determine joint position limits from env_ranges
        self.joints_low = [r[0] for r in self.env_ranges]
        self.joints_high = [r[1] for r in self.env_ranges]

        # (Joint effort limits are not used in this implementation.)
        self.joint_effort_low = [-0.15] * 18
        self.joint_effort_high = [0.15] * 18

        # Torso parameters
        self.torso_height_min = 0
        self.torso_height_max = 0.25
        self.torso_vx_min = -np.inf
        self.torso_vx_max = np.inf
        self.torso_w_roll_min = -np.inf
        self.torso_w_roll_max = np.inf
        self.torso_w_pitch_min = -np.inf
        self.torso_w_pitch_max = np.inf
        self.torso_w_yaw_min = -np.inf
        self.torso_w_yaw_max = np.inf
        self.torso_roll_min = -np.pi
        self.torso_roll_max = np.pi
        self.torso_pitch_min = -np.pi
        self.torso_pitch_max = np.pi
        self.torso_yaw_min = -np.pi
        self.torso_yaw_max = np.pi
        self.torso_y_min = -np.inf
        self.torso_y_max = np.inf

        # Foot contact thresholds
        self.rfs_min = 0
        self.rfs_max = 1
        self.lfs_min = 0
        self.lfs_max = 1

        # Build observation space by concatenating joint positions and additional states
        obs_low = np.append(
            self.joints_low,
            np.array([
                self.torso_height_min, self.torso_vx_min, self.torso_roll_min,
                self.torso_pitch_min, self.torso_yaw_min, self.torso_y_min,
                self.rfs_min, self.lfs_min
            ])
        )
        obs_high = np.append(
            self.joints_high,
            np.array([
                self.torso_height_max, self.torso_vx_max, self.torso_roll_max,
                self.torso_pitch_max, self.torso_yaw_max, self.torso_y_max,
                self.rfs_max, self.lfs_max
            ])
        )
        self.observation_space = spaces.Box(obs_low, obs_high)
        self.node.get_logger().debug("ACTION SPACES TYPE===>" + str(self.action_space))
        self.node.get_logger().debug("OBSERVATION SPACES TYPE===>" + str(self.observation_space))

        # Initialize sensor state variables
        self.torso_z = 0
        self.torso_y = 0
        self.torso_roll = 0
        self.torso_pitch = 0
        self.torso_yaw = 0
        self.torso_vx = 0
        self.torso_w_roll = 0
        self.torso_w_pitch = 0
        self.torso_w_yaw = 0

        # Create subscriptions for sensors (using the node provided by the parent)
        self.odom_subscriber = self.node.create_subscription(Odometry, '/plen/odom', self.odom_subscriber_callback, 10)
        self.joint_state_subscriber = self.node.create_subscription(JointState, '/plen/joint_states', self.joint_state_subscriber_callback, 10)
        self.right_contact_subscriber = self.node.create_subscription(ContactsState, '/plen/right_foot_contact', self.right_contact_subscriber_callback, 10)
        self.left_contact_subscriber = self.node.create_subscription(ContactsState, '/plen/left_foot_contact', self.left_contact_subscriber_callback, 10)
        self.right_contact = 1
        self.left_contact = 1

        # Call parent constructor (which may set up additional systems)
        super().__init__()
        self.node.get_logger().debug("END PlenWalkEnv INIT...")

    def odom_subscriber_callback(self, msg: Odometry):
        self.torso_z = msg.pose.pose.position.z
        self.torso_y = msg.pose.pose.position.y
        self.torso_x = msg.pose.pose.position.x
        quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        roll, pitch, yaw = euler_from_quaternion(quat)
        self.torso_roll = roll
        self.torso_pitch = pitch
        self.torso_yaw = yaw
        self.torso_vx = msg.twist.twist.linear.x
        self.torso_w_roll = msg.twist.twist.angular.x
        self.torso_w_pitch = msg.twist.twist.angular.y
        self.torso_w_yaw = msg.twist.twist.angular.z

    def joint_state_subscriber_callback(self, msg: JointState):
        joint_names = [
            'rb_servo_r_hip', 'r_hip_r_thigh', 'r_thigh_r_knee',
            'r_knee_r_shin', 'r_shin_r_ankle', 'r_ankle_r_foot',
            'lb_servo_l_hip', 'l_hip_l_thigh', 'l_thigh_l_knee',
            'l_knee_l_shin', 'l_shin_l_ankle', 'l_ankle_l_foot',
            'torso_r_shoulder', 'r_shoulder_rs_servo', 're_servo_r_elbow',
            'torso_l_shoulder', 'l_shoulder_ls_servo', 'le_servo_l_elbow'
        ]
        for i in range(len(joint_names)):
            idx = msg.name.index(joint_names[i])
            self.joint_poses[i] = msg.position[idx]
            self.joint_efforts[i] = msg.effort[idx]

    def right_contact_subscriber_callback(self, msg: ContactsState):
        # Compute total contact force magnitude from right foot contact sensor
        contact_force = Vector3()
        for state in msg.states:
            contact_force = state.total_wrench.force
        contact_force_np = np.array([contact_force.x, contact_force.y, contact_force.z])
        force_magnitude = np.linalg.norm(contact_force_np)
        self.right_contact = 1 if force_magnitude > 4.8559 / 3.0 else 0

    def left_contact_subscriber_callback(self, msg: ContactsState):
        contact_force = Vector3()
        for state in msg.states:
            contact_force = state.total_wrench.force
        contact_force_np = np.array([contact_force.x, contact_force.y, contact_force.z])
        force_magnitude = np.linalg.norm(contact_force_np)
        self.left_contact = 1 if force_magnitude > 4.8559 / 3.0 else 0

    def env_to_agent(self, env_range, env_val):
        agent_range = [-1, 1]
        m = (agent_range[1] - agent_range[0]) / (env_range[1] - env_range[0])
        b = agent_range[1] - m * env_range[1]
        agent_val = m * env_val + b
        return agent_val

    def agent_to_env(self, env_range, agent_val):
        agent_range = [-1, 1]
        m = (env_range[1] - env_range[0]) / (agent_range[1] - agent_range[0])
        b = env_range[1] - m * agent_range[1]
        env_val = m * agent_val + b
        if env_val >= env_range[1]:
            env_val = env_range[1] - 0.001
            self.node.get_logger().warn("Sampled Too High!")
        elif env_val <= env_range[0]:
            env_val = env_range[0] + 0.001
            self.node.get_logger().warn("Sampled Too Low!")
        return env_val

    def _set_init_pose(self):
        # Set simulation time for the next step (using rclpy.duration.Duration)
        self.next_sim_time = self.sim_time + rclpy.duration.Duration(seconds=self.running_step_sec)
        self.gazebo.unpauseSim()
        self.joints.set_init_pose(self.init_pose)
        self.gazebo.reset_joints(self.controllers_list, "plen")
        self.gazebo.pauseSim()
        time_to_iterate = self.next_sim_time - self.sim_time
        steps_to_iterate = (self.running_step - time_to_iterate.nanoseconds) * 1e-9 / self.gazebo._time_step
        if steps_to_iterate < 0:
            steps_to_iterate = 0
        else:
            self.iterate_proxy.call(int(steps_to_iterate))
        while self.sim_time < self.next_sim_time:
            pass

    def check_joints_init(self):
        joints_initialized = np.allclose(self.joint_poses, self.init_pose, atol=0.1, rtol=0)
        if not joints_initialized:
            self.node.get_logger().warn("Joints not all zero, trying again")
        else:
            self.node.get_logger().debug("All Joints Zeroed")
        return joints_initialized

    def _init_env_variables(self):
        # Initialize any variables needed at the start of an episode.
        pass

    def _set_action(self, action):
        env_action = np.empty(18)
        for i in range(len(action)):
            env_action[i] = self.agent_to_env(self.env_ranges[i], action[i])
        self.next_sim_time = self.sim_time + rclpy.duration.Duration(seconds=self.running_step_sec)
        self.gazebo.unpauseSim()
        self.joints.move_joints(env_action)
        self.gazebo.pauseSim()
        time_to_iterate = self.next_sim_time - self.sim_time
        steps_to_iterate = (self.running_step - time_to_iterate.nanoseconds) * 1e-9 / self.gazebo._time_step
        if steps_to_iterate < 0:
            steps_to_iterate = 0
        else:
            self.iterate_proxy.call(int(steps_to_iterate))
        while self.sim_time < self.next_sim_time:
            pass

    def _get_obs(self):
        observations = np.append(
            self.joint_poses,
            np.array([
                self.torso_z, self.torso_vx, self.torso_roll, self.torso_pitch,
                self.torso_yaw, self.torso_y, self.right_contact, self.left_contact
            ])
        )
        return observations

    def _is_done(self, obs):
        if (self.torso_roll > np.abs(np.pi / 3.) or 
            self.torso_pitch > np.abs(np.pi / 3.) or 
            self.torso_z < 0.08 or 
            self.torso_y > 1):
            done = True
            self.dead = True
        elif self.episode_timestep > self.max_episode_steps and self.torso_x < 1:
            done = True
            self.dead = False
        else:
            done = False
            self.dead = False
        return done

    def _compute_reward(self, obs, done):
        reward = 0
        reward += self.alive_reward
        reward += np.sign(self.torso_vx) * (self.torso_vx * self.vel_weight)**2
        reward -= (np.abs(self.init_height - self.torso_z) * self.height_weight)**2
        reward -= (np.abs(self.torso_y))**2 * self.straight_weight
        reward -= (np.abs(self.torso_roll))**2 * self.roll_weight
        reward -= (np.abs(self.torso_pitch))**2 * self.pitch_weight
        reward -= (np.abs(self.torso_yaw))**2 * self.yaw_weight
        if self.dead:
            reward -= self.dead_penalty
            self.dead = False
        return reward

