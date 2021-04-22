
'''
TD DDPG
'''

import torch
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal

import threading as td
from multiprocessing.managers import BaseManager
from multiprocessing import Process, Manager
from torch.multiprocessing import Process
import torch.multiprocessing as mp

import matplotlib.pyplot as plt

import time
import argparse
import math
# import random
import os

import gym
import gym_gazeboros_ac
import numpy as np
# from logger import Logger


# critical for make multiprocessing work
torch.multiprocessing.set_start_method('forkserver', force=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device is {device}')


###############################  hyper parameters  #########################

ENV_NAME = 'gazeborosAC-v0'  # 'gazeborosAC-v0'  # environment name
RANDOMSEED = 42  # random seed

###############################  TD DDPG   #################################


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                 name, chkpt_dir='./model_weights/TD3'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')
        self.checkpoint_dir_periodic = chkpt_dir+'/periodic'

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        print('\n... saving checkpoint ...\n')
        T.save(self.state_dict(), self.checkpoint_file)

    def save_periodic_checkpoint(self, epoch):
        print('\n... saving periodic checkpoint ...\n')
        T.save(self.state_dict(), os.path.join(
            self.checkpoint_dir_periodic, f'{self.name}_td3_{epoch:.0f}'))

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir='./model_weights/TD3'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_dir_periodic = chkpt_dir+'/periodic'

        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = T.tanh(self.mu(prob))

        return mu

    def save_checkpoint(self):
        print('\n... saving checkpoint ...\n')
        T.save(self.state_dict(), self.checkpoint_file)

    def save_periodic_checkpoint(self, epoch):
        print('\n... saving periodic checkpoint ...\n')
        T.save(self.state_dict(), os.path.join(
            self.checkpoint_dir_periodic, f'{self.name}_td3_{epoch:.0f}'))

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
                 gamma=0.99, update_actor_interval=2, warmup=1000,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=100, noise=0.1):
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions, name='actor')

        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions, name='critic_2')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, n_actions=n_actions, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, n_actions=n_actions, name='target_critic_2')

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise,
                                           size=(self.n_actions,)), device=self.device)
        else:
            state = T.tensor(observation, dtype=T.float, device=self.device)
            mu = self.actor.forward(state).to(self.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),
                                 dtype=T.float, device=self.device)

        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        # seconds = time.time()

        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward,    dtype=T.float, device=self.device)
        done = T.tensor(done,                     device=self.device)
        state_ = T.tensor(new_state, dtype=T.float, device=self.device)
        state = T.tensor(state,     dtype=T.float, device=self.device)
        action = T.tensor(action,    dtype=T.float, device=self.device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + \
            T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action[0],
                                 self.max_action[0])

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        # print(f'Time to update_network_parameters is: {time.time() - seconds}\n\n')

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + \
                (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def save_periodic_models(self, epoch):
        self.actor.save_periodic_checkpoint(epoch)
        self.target_actor.save_periodic_checkpoint(epoch)
        self.critic_1.save_periodic_checkpoint(epoch)
        self.critic_2.save_periodic_checkpoint(epoch)
        self.target_critic_1.save_periodic_checkpoint(epoch)
        self.target_critic_2.save_periodic_checkpoint(epoch)

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()


###############################  MISC   ####################################


class PolicyNetwork(nn.Module):
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample()
        action = mean+std*z
        action = torch.clamp(action, -self.action_range, self.action_range)
        return action.squeeze(0)

    def sample_action(self,):
        a = torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return a.numpy()


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


def plot(rewards, FILENAME):
    plt.figure(figsize=(10, 5))

    running_avg = np.zeros(len(rewards))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(rewards[max(0, i-50):(i+1)])

    plt.plot(rewards, label='Rewards')
    plt.plot(running_avg, label='Running Avg')
    plt.title('Running average of previous 50 rewards')
    plt.legend()
    plt.savefig(FILENAME)
    # plt.show()
    # plt.clf()

    # # plt.savefig(figure_file)
    # plt.show()

###############################  MAIN   ####################################


def main():
    # reproducible
    # env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    torch.manual_seed(RANDOMSEED)

    env = NormalizedActions(gym.make(ENV_NAME).unwrapped)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f'State_dim:  {state_dim}')
    print(f'Action_dim: {action_dim}')

    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)

    agent = Agent(alpha=0.001, beta=0.001,
                  input_dims=[state_dim], tau=0.005,
                  env=env, batch_size=100, layer1_size=400, layer2_size=300,
                  n_actions=action_dim)
    agent.load_models()



    f = open("./model_weights/TD3/logs.txt", "w")
    score_history = []

    n_games = 1400  # all night
    n_games = 700   # half night
    window = 25
    best_score = 5  # env.reward_range[0]# this does not work as intended
    print(f'best_score: {best_score}')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            # if done: break # cleaner exit, but might not learn terminal state...
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-window:])

        if i % 500 == 0:
            agent.save_periodic_models(epoch=i)

        if avg_score > best_score and i > window:
            # print(f'avg_score is greater than best_score... Saving Model')
            plot(score_history,
                 f'Turtlebot_Continuous_TD_DDPG_games_{n_games}.png')
            best_score = avg_score
            agent.save_models()

        f.write(f"Episode {i} Score {score:.1f} Average score {avg_score:.1f}\n")
        print(f'Episode {i} Score {score:.1f} Average score {avg_score:.1f}\n\n')

    f.close()
    env.close()
    plot(score_history, f'Turtlebot_Continuous_TD_DDPG_games_{n_games}.png')
    # if n_games >= 1000: # rather than best model
    agent.save_periodic_models(epoch=n_games)

    # a = ppo.choose_action(s)
    # a = [-0.91, -0.91]
    # done = False
    # while not done:
    #     s_, r, done, _ = env.step(a)

    # if done:
    #     # break
    #     print(f'Exiting..')
    #     env.close()
    #     exit()

    # print(f'Episode: {ep}/{EP_MAX}  | Episode Reward: {ep_r:.4f}  | Running Time: {time.time() - t0:.4f}')
    # logger.scalar_summary("reward".format(id), ep_r, ep*4)

    # ppo.save_model(MODEL_PATH)
    # env.close()


if __name__ == '__main__':
    main()
