#!/usr/bin/env python3
"""
Converted from ROS1 to ROS2 (Humble on Ubuntu 22.04)
File: td3.py

Contains the Twin Delayed Deep Deterministic Policy Gradient (TD3)
agent, along with Actor, Critic, ReplayBuffer, and evaluation/training routines.
"""

import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import os

# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        return self.max_action * torch.tanh(self.fc3(a))

# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        return self.fc3(q1)

# --- Replay Buffer ---
class ReplayBuffer(object):
    def __init__(self, max_size=1000000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        my_path = os.path.abspath(os.path.dirname(__file__))
        self.buffer_path = os.path.join(my_path, "../../replay_buffer")

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def save(self, iterations):
        if not os.path.exists(self.buffer_path):
            os.makedirs(self.buffer_path)
        with open(os.path.join(self.buffer_path, f'replay_buffer_{iterations}.data'), 'wb') as filehandle:
            pickle.dump(self.storage, filehandle)

    def load(self, iterations):
        with open(os.path.join(self.buffer_path, f'replay_buffer_{iterations}.data'), 'rb') as filehandle:
            self.storage = pickle.load(filehandle)

    def sample(self, batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []
        for i in ind:
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))
        state = torch.FloatTensor(np.array(states)).to(device)
        action = torch.FloatTensor(np.array(actions)).to(device)
        next_state = torch.FloatTensor(np.array(next_states)).to(device)
        reward = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        not_done = torch.FloatTensor(1. - (np.array(dones).reshape(-1, 1))).to(device)
        return state, action, next_state, reward, not_done

# --- TD3 Agent ---
class TD3Agent(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=self.device))
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=self.device))


def evaluate_policy(policy, env_name, seed, eval_episodes=10, render=False):
    eval_env = gym.make(env_name, render=render)
    eval_env.seed(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print("Evaluation over {} episodes: {}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    if render:
        eval_env.close()
    return avg_reward


def trainer(env_name, seed, max_timesteps, start_timesteps, expl_noise,
            batch_size, eval_freq, save_model, file_name="best_avg"):
    if not os.path.exists("../results"):
        os.makedirs("../results")
    if not os.path.exists("../models"):
        os.makedirs("../models")
    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy = TD3Agent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    evaluations = [evaluate_policy(policy, env_name, seed, 1)]
    state = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    for t in range(int(max_timesteps)):
        episode_timesteps += 1
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (policy.select_action(np.array(state)) +
                      np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(-max_action, max_action)
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        replay_buffer.add((state, action, next_state, reward, done_bool))
        state = next_state
        episode_reward += reward
        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)
        if done:
            print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(
                t + 1, episode_num, episode_timesteps, episode_reward))
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        if (t + 1) % eval_freq == 0:
            evaluations.append(evaluate_policy(policy, env_name, seed, 1))
            np.save("../results/" + str(file_name) + str(t), evaluations)
            if save_model:
                policy.save("../models/" + str(file_name) + str(t))


if __name__ == "__main__":
    trainer("BipedalWalker-v2", 0, 1e6, 1e4, 0.1, 100, 15e3, True)

