'''
Multi-processing for PPO continuous version 1
'''

import math
import random

import gym
import gym_gazeboros#_ac
import numpy as np
# from logger import Logger

import torch
# torch.multiprocessing.set_start_method('forkserver', force=True) # critical for make multiprocessing work
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display

import argparse
import time

import torch.multiprocessing as mp
from torch.multiprocessing import Process

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

import threading as td


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#####################  hyper parameters  ####################

ENV_NAME = 'gazeboros-v0'  # environment name
RANDOMSEED = 2  # random seed
PROJECT_NAME = "ali's test"  # Project name for loging

EP_MAX = 100000  # total number of episodes for training
EP_LEN = 80  # total number of steps for each episode
GAMMA = 0.95  # reward discount
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
BATCH = 256  # update batchsize
A_UPDATE_STEPS = 4  # actor update steps
C_UPDATE_STEPS = 4  # critic update steps
EPS = 1e-8   # numerical residual
MODEL_PATH = 'model/ppo_multi'
NUM_WORKERS = 1  # or: mp.cpu_count()
ACTION_RANGE = 1.  # if unnormalized, normalized action range should be 1.
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.1),  # Clipped surrogate objective, find this is better
    ][1]  # choose the method for optimization
###############################  PPO  ####################################

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
            n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc2_5 = nn.Linear(self.fc2_dims, int(self.fc2_dims/2))
        self.fc3 = nn.Linear(int(self.fc2_dims/2), self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = device # torch.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(self.device)
        self.to(self.device)

    def forward(self, state):
        state = state.float()
        #print(state.dtype)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_5(x))
        actions = self.fc3(x)

        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
            max_mem_size=100000, eps_end=0.05, eps_dec=5e-4, ALIs_over_training=2):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 30
        self.ALIs_over_training = ALIs_over_training

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=512, fc2_dims=256)
        self.Q_next = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=512, fc2_dims=256) # 64 ,64, if not updating pramas

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size: #maybe self.batch_size*2... IDK about this 
            return

        
        
        max_mem = min(self.mem_cntr, self.mem_size)

        # replace=False means dont given duplicates. max_mem isnt mutated
        batch = np.random.choice(max_mem, self.batch_size, replace=False) # todo decrease and force train on last 3 
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        #N = 2 if self.iter_cntr > self.batch_size else 1 # maybe use self.mem_cntr
        for i in range(self.ALIs_over_training): # Ali over training 
            self.Q_eval.optimizer.zero_grad()
            q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
            q_next = self.Q_eval.forward(new_state_batch)
            q_next[terminal_batch] = 0.0

            q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()

            self.iter_cntr += 1
            
            if self.iter_cntr % self.replace_target == 0:
                self.Q_next.load_state_dict(self.Q_eval.state_dict())
                
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
        # This isn't my code. IDK why we dont optimize Q_next, however, I trust the author (youtube: machine learning with Phil). 
        # This was because the two networks are different... IDK how to update the Q_next network

def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.savefig('ppo_multi.png')
    # plt.show()
    plt.clf()


def main():
    # reproducible
    np.random.seed(RANDOMSEED)
    torch.manual_seed(RANDOMSEED)

    env = NormalizedActions(gym.make(ENV_NAME).unwrapped)
    env.seed(RANDOMSEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(env.action_space)      # 2
    print(env.observation_space) # 47
    action_space = 2
    observation_space = 47

    ########################
    observation = env.reset()
    exit()
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=action_space, eps_end=0.01,
                input_dims=[observation], lr=0.001, eps_dec=5e-4*1.1, ALIs_over_training=1) # changed from eps_dec=5e-4

    scores, eps_history = [], []
    n_games = 150

    start = time.time()
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        

        if i % 1 == 0:
            avg_score = np.mean(scores[-10:])
            print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)

    end = time.time()
    print(f'Time taken: {(end - start):.4f}')



    # if args.test:
    #     while True:
    #         s = env.reset()
    #         for i in range(EP_LEN):
    #             env.render()
    #             s, r, done, _ = env.step(ppo.choose_action(s))
    #             if done:
    #                 break
if __name__ == '__main__':
    main()
