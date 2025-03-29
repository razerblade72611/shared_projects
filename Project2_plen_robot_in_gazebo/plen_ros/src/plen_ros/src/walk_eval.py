#!/usr/bin/env python3

import os
import time
import numpy as np
import gym
import torch
import rclpy
from rclpy.node import Node

from plen_ros_helpers.td3 import ReplayBuffer, TD3Agent, evaluate_policy
from plen_ros_helpers import plen_walk

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('plen_td3')
    node.get_logger().info("STARTING PLEN_TD3 NODE")

    # TRAINING PARAMETERS
    env_name = "PlenWalkEnv-v0"
    seed = 0
    max_timesteps = int(4e6)

    # Find absolute path to this file
    my_path = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(my_path, "../results")
    models_path = os.path.join(my_path, "../models")

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    env = gym.make(env_name)

    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    node.get_logger().warn("RECORDED MAX ACTION: {}".format(max_action))

    policy = TD3Agent(state_dim, action_dim, max_action)
    # Optionally load existing policy; replace 9999 with the appropriate number.
    policy_num = 1989999
    policy_file = os.path.join(models_path, "plen_walk_gazebo_" + str(policy_num) + "_critic")
    if os.path.exists(policy_file):
        node.get_logger().info("Loading Existing Policy")
        policy.load(os.path.join(models_path, "plen_walk_gazebo_" + str(policy_num)))

    replay_buffer = ReplayBuffer()
    # Optionally load an existing replay buffer; change buffer_number if needed.
    buffer_number = 0  # By default will load nothing; change this as necessary.
    replay_buffer_file = os.path.join(replay_buffer.buffer_path, "replay_buffer_" + str(buffer_number) + '.data')
    if os.path.exists(replay_buffer_file):
        node.get_logger().info("Loading Replay Buffer " + str(buffer_number))
        replay_buffer.load(buffer_number)
        node.get_logger().info(str(replay_buffer.storage))

    # Initialize evaluations list.
    evaluations = []

    state = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    node.get_logger().info("STARTED PLEN_TD3 NODE")

    for t in range(max_timesteps):
        episode_timesteps += 1
        # Deterministic policy action (clipped)
        action = np.clip(policy.select_action(np.array(state)),
                         0.99 * -max_action, 0.99 * max_action)

        # Perform action in the environment.
        next_state, reward, done, _ = env.step(action)

        state = next_state
        episode_reward += reward

        if done:
            node.get_logger().info(
                "Total T: {} Episode Num: {} Episode T: {} Reward: {}"
                .format(t + 1, episode_num, episode_timesteps, episode_reward)
            )
            # Reset environment for the next episode.
            state = env.reset()
            done = False
            evaluations.append(episode_reward)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

