#!/usr/bin/env python3
"""
Converted from ROS1 to ROS2 (Humble on Ubuntu 22.04)
File: robot_gazebo_env.py

This is the base class for robot Gazebo environments. It uses rclpy
to subscribe to the simulation clock, create service clients (e.g. for /iterate),
and publish rewards. Extend this class by implementing the extension methods.
"""

import rclpy
from rclpy.node import Node
import gym
from gym.utils import seeding
import numpy as np
import time

from rosgraph_msgs.msg import Clock

# Import the converted Gazebo and controllers connections (assumed converted separately)
from plen_ros_helpers.gazebo_connection import GazeboConnection
from plen_ros_helpers.controllers_connection import ControllersConnection

# ROS2 message and service types (ensure that these interfaces have been ported)
from plen_ros_msgs.msg import RLExperimentInfo  # Adjust package name if needed
from plen_ros_srvs.srv import Iterate          # Adjust package name if needed


class RobotGazeboEnv(gym.Env):
    def __init__(self,
                 robot_name_space,
                 controllers_list,
                 reset_controls,
                 start_init_physics_parameters=True,
                 reset_world_or_sim="SIMULATION"):
        """
        Base class for Gazebo-based Gym environments.
        Initializes connections to Gazebo, controllers, clock subscriber,
        and the iterate service for deterministic stepping.
        """
        # Create an internal node for ROS2 operations.
        rclpy.init(args=None)
        self.node = rclpy.create_node('robot_gazebo_env')

        self.node.get_logger().debug("START init RobotGazeboEnv")

        # Create Gazebo connections
        self.gazebo = GazeboConnection(self.node,
                                       start_init_physics_parameters,
                                       reset_world_or_sim)
        self.gazebo_sim = GazeboConnection(self.node,
                                           start_init_physics_parameters,
                                           "SIMULATION")
        self.controllers_object = ControllersConnection(
            self.node, namespace=robot_name_space, controllers_list=controllers_list)
        self.reset_controls = reset_controls
        self.controllers_list = controllers_list
        self.robot_name_space = robot_name_space

        # Seed the environment using gym's seeding
        self.np_random, _ = seeding.np_random(None)

        # Subscribe to the simulation clock
        self.clock_subscriber = self.node.create_subscription(
            Clock, '/clock', self.clock_callback, 10)

        # Create a service client for the /iterate service.
        self.iterate_client = self.node.create_client(Iterate, '/iterate')
        if not self.iterate_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error("Service /iterate not available!")

        # Initialize episode variables
        self.episode_num = 0
        self.cumulated_episode_reward = 0
        self.episode_timestep = 0
        self.total_timesteps = 0

        # Set up moving average reward publisher
        self.moving_avg_buffer_size = 1000
        self.moving_avg_buffer = np.zeros(self.moving_avg_buffer_size)
        self.moving_avg_counter = 0
        self.reward_pub = self.node.create_publisher(
            RLExperimentInfo, '/' + self.robot_name_space + '/reward', 1)

        # Unpause simulation to allow data flow and check systems
        self.gazebo.unpauseSim()
        # Optionally, reset controllers:
        # self.controllers_object.reset_controllers()

        self._check_all_systems_ready()
        self.gazebo.pauseSim()

        self.node.get_logger().debug("END init RobotGazeboEnv")

    def clock_callback(self, data):
        """
        Callback for the simulation clock.
        Updates self.sim_time with the latest simulation time.
        """
        self.sim_time = data.clock

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Executes one time step by applying the action, obtaining observations,
        computing reward, and checking termination.
        """
        self._set_action(action)  # Extension method; must be implemented in child class.
        obs = self._get_obs()     # Extension method.
        done = self._is_done(obs) # Extension method.
        info = {}
        reward = self._compute_reward(obs, done)  # Extension method.
        self.cumulated_episode_reward += reward
        self.episode_timestep += 1
        self.total_timesteps += 1
        return obs, reward, done, info

    def reset(self):
        """
        Resets the simulation and environment variables.
        """
        self._reset_sim()
        self._init_env_variables()  # Extension method.
        self._update_episode()
        obs = self._get_obs()       # Extension method.
        return obs

    def close(self):
        """
        Cleanly shuts down the environment.
        """
        self.node.get_logger().error("Closing RobotGazeboEnvironment")
        self.node.destroy_node()
        rclpy.shutdown()

    def _update_episode(self):
        """
        Publishes the cumulative reward for the episode and resets counters.
        """
        self._publish_reward_topic(self.cumulated_episode_reward, self.episode_num)
        self.episode_num += 1
        self.moving_avg_counter += 1
        self.cumulated_episode_reward = 0
        self.episode_timestep = 0

    def _publish_reward_topic(self, reward, episode_number=1):
        """
        Publishes an RLExperimentInfo message with reward and episode info.
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.total_timesteps = self.total_timesteps
        reward_msg.episode_reward = reward

        # Update moving average buffer
        if self.moving_avg_counter >= self.moving_avg_buffer_size:
            self.moving_avg_counter = 0
        self.moving_avg_buffer[self.moving_avg_counter] = self.cumulated_episode_reward
        if self.episode_num >= self.moving_avg_buffer_size:
            reward_msg.moving_avg_reward = float(np.average(self.moving_avg_buffer))
        else:
            reward_msg.moving_avg_reward = float(np.nan)
        self.reward_pub.publish(reward_msg)

        if np.isnan(reward):
            self.node.get_logger().error("Reward is NaN; closing simulation")
            self.close()

    def _reset_sim(self):
        """
        Resets the simulation. If reset_controls is True, resets controllers and
        sets joints to initial pose.
        """
        self.node.get_logger().debug("RESETTING simulation")
        if self.reset_controls:
            self.node.get_logger().debug("RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self._set_init_pose()  # Extension method.
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self._set_init_pose()
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()
        else:
            self.node.get_logger().debug("Not resetting controllers")
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()
        self.node.get_logger().debug("RESET SIM END")
        return True

    # --- Extension methods (to be implemented by child classes) ---
    def _set_init_pose(self):
        raise NotImplementedError()

    def _check_all_systems_ready(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _init_env_variables(self):
        raise NotImplementedError()

    def _set_action(self, action):
        raise NotImplementedError()

    def _is_done(self, observations):
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        raise NotImplementedError()

