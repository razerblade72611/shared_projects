#!/usr/bin/env python3

import os
import time
import numpy as np
import gym
import torch
import rclpy
from rclpy.node import Node

# Import your custom TD3 agent, replay buffer, and evaluation function.
from plen_ros_helpers.td3 import ReplayBuffer, TD3Agent, evaluate_policy
from plen_ros_helpers import plen_walk  # if used

class PlenTD3(Node):
    def __init__(self):
        super().__init__('plen_td3')
        self.get_logger().info("STARTING PLEN_TD3 NODE")

        # TRAINING PARAMETERS
        self.env_name = "PlenWalkEnv-v0"
        self.seed = 0
        self.max_timesteps = int(4e6)
        self.start_timesteps = int(1e4)  # use 1e3 for testing if needed
        self.expl_noise = 0.1
        self.batch_size = 100
        self.eval_freq = int(1e4)
        self.save_model = True
        self.file_name = "plen_walk_gazebo_"

        # Determine paths for results and models.
        my_path = os.path.abspath(os.path.dirname(__file__))
        self.results_path = os.path.join(my_path, "../results")
        self.models_path = os.path.join(my_path, "../models")
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)

        # Create gym environment.
        self.env = gym.make(self.env_name)

        # Set seeds.
        self.env.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.get_logger().warn("RECORDED MAX ACTION: {}".format(self.max_action))

        self.policy = TD3Agent(self.state_dim, self.action_dim, self.max_action)
        self.replay_buffer = ReplayBuffer()
        buffer_number = 9999  # Change this value if needed.
        buffer_file = os.path.join(self.replay_buffer.buffer_path, f"replay_buffer_{buffer_number}.data")
        if os.path.exists(buffer_file):
            self.get_logger().info("Loading Replay Buffer " + str(buffer_number))
            self.replay_buffer.load(buffer_number)

        self.evaluations = []
        self.get_logger().info("STARTED PLEN_TD3 NODE")

    def run(self):
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(self.max_timesteps):
            episode_timesteps += 1

            # Select action: random for initial timesteps, else policy + exploration noise.
            if t < self.start_timesteps:
                action = self.env.action_space.sample()
            else:
                action = (
                    self.policy.select_action(np.array(state)) +
                    np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                )
                # Clip the action to avoid reaching limits.
                action = np.clip(action, -0.99 * self.max_action, 0.99 * self.max_action)

            next_state, reward, done, _ = self.env.step(action)
            done_bool = float(done) if episode_timesteps < self.env._max_episode_steps else 0

            # Store the transition in the replay buffer.
            self.replay_buffer.add((state, action, next_state, reward, done_bool))
            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data.
            if t >= self.start_timesteps:
                self.policy.train(self.replay_buffer, self.batch_size)

            if done:
                state = self.env.reset()
                self.evaluations.append(episode_reward)
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Periodically save evaluations and model.
            if (t + 1) % self.eval_freq == 0:
                np.save(os.path.join(self.results_path, self.file_name), self.evaluations)
                if self.save_model:
                    self.policy.save(os.path.join(self.models_path, self.file_name + str(t)))
                self.get_logger().info("Saved model and evaluations at timestep: {}".format(t + 1))


def main(args=None):
    rclpy.init(args=args)
    plen_td3 = PlenTD3()
    try:
        plen_td3.run()
    except KeyboardInterrupt:
        plen_td3.get_logger().info("Training interrupted by user.")
    finally:
        plen_td3.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

