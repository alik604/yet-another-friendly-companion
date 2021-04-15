import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.autograd import Variable

from torch import nn, optim, distributions

import numpy as np
import gym
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy

import torchvision.transforms as T
#from typing import Tuple
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)

use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

###############################  hyper parameters  #########################
#ENV_NAME = 'gazeboros-v0' # 'gazeborosAC-v0'  # environment name
ENV_NAME = 'LunarLander-v2'


RANDOMSEED = 42  # random seed
 
############################################################################
LR = 0.001 
reward = 1
#gaussian = 5


def gmm_loss(batch, mus, sigmas, logpi, reduce:bool = True):
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob

class RNN(nn.Module):
    def __init__(self, obs_dim, act_dim, gaussian, hid_dim=64, drop_prob = 0.5):
        super(RNN, self).__init__()
        self.action_dim = act_dim
        self.observation_dim = obs_dim 
        self.hidden = hid_dim
        self.gaussian_mix = gaussian
        self.reward = reward
        self.learning_r = LR 
        gmm_out = (2*obs_dim+1) *gaussian + 2

        self.rnn = nn.LSTMCell(obs_dim+act_dim, hid_dim) 
        #self.dropout = nn.Dropout(drop_prob)
        #self.fc = nn.Linear(hid_dim, gmm_out)
        #self.sigmoid = nn.Sigmoid()
        self.mu = nn.Linear(hid_dim, obs_dim)
        self.logsigma = nn.Linear(hid_dim, obs_dim)
    
    def forward(self, obs, act, hid=64):

        print("the observations are:", obs)
        print("the actions are", act)
        '''
        print(hid)
        x = torch.cat([act, obs], dim=-1)
        print("the concatenated result:", x)
        #exit(-1)
        lstm_out, hidden = self.rnn(x, hid)
        gmm_outs = self.fc(lstm_out)

        stride = self.gaussian_mix * self.observation_dim

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, batch_size, self.gaussian_mix, self.observation_dim)
        
        sigma = gmm_outs[:, :, stride:2 * stride]
        #sigma = sigmas.view(seq_len, batch_size, self.gaussian_mix, self.observation_dim)
        sigma = torch.exp(sigma)

        pi = gmm_outs[:, :, 2 *stride: 2 *stride +self.gaussian_mix]
        #pi = pi.view(seq_len, batch_size, self.gaussian_mix)
        logpi = F.log_softmax(pi, dim=-1)

        return mus, sigma, logpi
        '''
        #print("the action is:, ", torch.from_numpy(np.array([act, 0])))
        #act = torch.tensor([act])
        #print("the state is :, ", obs)
        x = torch.cat([act, obs], dim=-1)
        h, c = self.rnn(x, hid)
        mu = self.mu(h)
        sigma = torch.exp(self.logsigma(h))
        return mu, sigma, (h, c)
    
        #lstm_out = lstm_out.contiguous().view(-1, self.hidden) #TODO check if this is needed

        #out = self.dropout(lstm_out)
        #
        # out = self.fc(out)
        #out = self.sigmoid(out)

        #out = out.view(batch_size, -1)
        #out = out[:, -1]
        #mu = self.mu(h)
        #sigma = torch.exp(self.logsigma(h))
        #return mu, sigma, (h, c)
        #return out, hid


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

np.random.seed(0)
torch.manual_seed(0)

env =gym.make(ENV_NAME)
#env = gym.make('CartPole-v0').unwrapped

#print(env)
env.seed(0)
obs_dim = env.observation_space.shape[0] #this is for the lunar lander environment
#act_dim = env.action_space.shape[0]
act_dim = env.action_space.n

print("obs_dim + act_dim", obs_dim+act_dim)
#print("Initializing agent (device=" +  str(device)  + ")...")
model = RNN(obs_dim, act_dim, 5)

model.zero_grad()
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = optim.RMSprop(model.parameters())
criterion = torch.nn.MSELoss()
#memory = ReplayMemory(10000)




#optimizer = optim.RMSprop(model.parameters())
#memory = ReplayMemory(10000)


batch_size = 1
ls = []
losses = []
num_episodes = 10
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    hid = (torch.zeros(batch_size, model.hidden).to(device),torch.zeros(batch_size, model.hidden).to(device))
    #state = torch.from_numpy(np.array([state]))
    ls.append(state)
    loss = 0.0
    obs_batch, next_obs_batch = ls[:-1],ls[1:]
    for t in count():
        # Select and perform an action
        #need to get state data and make a sequence of the state data 
        #zero pre-padding 
        '''
        np.array([
            np.array(state_at_a_time_step])
            ,np.array(state_at_a_time_step])
            ,np.array(state_at_a_time_step])
            ,np.array(state_at_a_time_step])
            ,np.array(state_at_a_time_step])
            ,np.array(state_at_a_time_step])
            ,np.array(state_at_a_time_step])
            ,np.array(state_at_a_time_step]) #current one should be very last timestep 
            ])
        '''
        #seq_len = look back window
        #       #if len(state_array)== 10:
        #    then pred = model.forward(state_array, action, hid )
        #else: 
        #    state 
        #    action = [0]
        #ls = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        #env = new env 

       # while True:
       #     state = env.step()
       #     ls.append(state)

       # input = ls[:-10] # 9 zeros, and new state #look at engineer man on youtube!!!! sentence generation and lyric generation

       # lstm(input)


       # if env is done:
        #    break
        #LSTM expects 3D input
        state = torch.from_numpy(np.array([state]))
        
        ls.append(state)
        print("nextobservation looks like:,", ls[1:])
        next_obs = ls[1:]
        next_obs = next_obs[0]

        act_batch = len(ls)
        #action = env.action_space.sample()
        #action = torch.stack(action).to(device)
        action = torch.tensor([[0,0 ,0 ,0]])
        print(action)
        mu, sigma, hid = model.forward(state, action, hid)
        print("this is what mu looks like:", mu)
        print("this is what sigma looks like:", sigma)
        dist = distributions.Normal(loc=mu, scale=sigma)
        nll = -dist.log_prob(next_obs) # negative log-likelihood
        nll = torch.mean(nll, dim=-1)     # mean over dimensions
        nll = torch.mean(nll, dim=0)      # mean over batch
        loss += nll
        loss = loss / len(ls)     # mean over trajectory
        val = loss.item()

        state = ls[:-1][0]
    losses.append(val)

print("idk")
#next_action = select_action(state, action, hid)
#next_state, reward, done, _ = env.step(action)
#reward = Tensor([reward])

# Store the transition in memory
#memory.push(state, action, next_state, reward)

# Move to the next state
#state = next_state

# Perform one step of the optimization (on the target network)
#optimize_model()
'''
if done:
    print('Episode %02i, Duration %i' % (i_episode, t+1))
    episode_durations.append(t + 1)
    plot_durations()
    break
'''
