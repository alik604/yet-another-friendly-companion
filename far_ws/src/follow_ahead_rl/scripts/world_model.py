import math
import random

import gym
# from gym_gazeboros_ac import gym_gazeboros_ac
import gym_gazeboros_ac
import numpy as np
from logger import logger
from PIL import Image
from collections import namedtuple 
from itertools import count 

import torch
torch.multiprocessing.set_start_method('forkserver', force=True) # critical for make multiprocessing work
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Normal, MultivariateNormal

from IPython.display import clear_output

import matplotlib
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


#reference: https://github.com/ctallec/world-models/blob/master/models/mdrnn.py
#set up matplotlib ->>>>>. important: note sure if we need this section for GAZEBO
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display 

plt.ion()

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()

#####################  hyper parameters  ####################


ENV_NAME = 'gazeborosAC-v0'  # environment name

RANDOMSEED = 2  # random seed
PROJECT_NAME = "ppo_v_0.2_2point"  # Project name for loging

EP_MAX = 20000  # total number of episodes for training
EP_LEN = 80  # total number of steps for each episode
GAMMA = 0.999  # reward discount
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
BATCH_SIZE = 128  # update batchsize
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

'''

A_UPDATE_STEPS = 4  # actor update steps
C_UPDATE_STEPS = 4  # critic update steps
EPS = 1e-8   # numerical residual
MODEL_PATH = 'model/ppo_multi'
NUM_WORKERS = 4  # or: mp.cpu_count()
ACTION_RANGE = 1.  # if unnormalized, normalized action range should be 1.
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.1),  # Clipped surrogate objective, find this is better
    ][1]  # choose the method for optimization

'''
def gmm_loss(batch, mus, sigmas, logpi, reduce=True): # pylint: disable=too-many-arguments
    """ Computes the gmm loss.
    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.
    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited
    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))
    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
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

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity 
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """save a transition"""

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = Transition(*args)
        self.position = (Self.position +1) % self.capacity 
    

    def sample (self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, *inputs):
        pass
#Q-network: a CNN that takes in the difference btween the current and previous screen patches. 
#Two ouputs: Q(s, LEFT) and Q(s, RIGHT) -- s is input into network
#try to predict expected return of taking each action given the curren input
class ActionNet(_MDRNNBase):

    def __init__(self, latents, actions, hiddens, gaussians):
	super().__init__(latents, actions, hiddens, gaussians)
	self.rnn = nn.LSTM(latents+ actions, hiddens)
	
	

    #call with either one element to determine next action or a batch during optimization
    def forward(self, actions, latents):
 	""" MULTI STEPS forward.
        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
	seq_len, bs = actions.size(0), actions.size(1)
	ins = torch.cat([actions, latents], dim = -1)
	outs, _ = self.rnn(ins)
	gmm_outs = self.gmm_linear(outs)
	
	stride = self.gaussians * self.latents
	mus = gmm_outs[:, :, :stride]
	mus = mus.view(seq_len, bs, self.gaussians, self.latents)
	
	sigmas = gmm_outs[:, :, stride:2 * stride]
	sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
	sigmas = torch.exp(sigmas)

	pi = gmm_outs[:, :, 2*stride:2*stride+ self.gaussians]
	pi = pi.view(seq_len, bs, self.gaussians)
	logpi = f.log_softmax(pi, dim = -1)

	rs = gmm_outs[:, :, -2]
	
	ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds

class MDRNNCell(_MDRNNBase):
    """ MDRNN model for one step forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden): # pylint: disable=arguments-differ
        """ ONE STEP forward.
        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: (BSIZE, LSIZE) torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        """
        in_al = torch.cat([action, latent], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden

def select_action(state):
    state = torch.FloatTensor(state).to(device)

    # TODO: this function

    # global steps_done
    # sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * steps_done / EPS_DECAY)
    # steps_done += 1
    # if sample > eps_threshold:
    #     with torch.no_grad():
    #         # t.max(1) will return largest column value of each row.
    #         # second column on max result is index of where max element was
    #         # found, so we pick action with the larger expected reward.
    #         return action_net(state).max(1)[1].view(1, 1)
    # else:
    #     return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to action_net
    state_action_values = action_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in action_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

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

def translate_state(state):
    state = np.ndarray.tolist(state)
    new_state = [[], [], [], []]
    new_state[0] = state[0:12]
    new_state[1] = state[12:24]
    new_state[2] = state[24:36]
    new_state[3] = state[36:47]

    new_state[3].append(0)

    return new_state


# MAIN " "

np.random.seed(RANDOMSEED)
torch.manual_seed(RANDOMSEED)

env = NormalizedActions(gym.make(ENV_NAME).unwrapped)
env.set_agent(0)

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

action_net = ActionNet(4, 12, n_actions).to(device)

optimizer = optim.RMSprop(action_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward'))

# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation = 'none')
# plt.title('Example extracted screen')
# plt.show()
episode_durations = []
num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()

    state = translate_state(state)


    for t in count():
        # Select and perform an action
        action = select_action(state)
        print(action)
        
        next_state, reward, done, _ = env.step(action)

        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        value_net.load_state_dict(action_net.state_dict())

print('Complete')
# ENV_NAME.render()
env.close()
# plt.ioff()
# plt.show()
