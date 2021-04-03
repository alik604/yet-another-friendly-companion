#1. selection: starts at rood node R then moves down the tree  by selecting optimal child nodes until a leaf node L is reached
#2. Expansion: if L is not a terminal node then create one or more child nodes according to available actions at current state,
#  then select the first of these new nodes
#3. Simulation: Run a simulated rollout from M until terminal state is found
#4. backpropogation
# Output: goal (x, y position) or velocity -- goals as edges, pick reward function

import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from tqdm import trange
import seaborn as sns
import pandas as pd
from random import random
import warnings
import torch
import torch as T
torch.multiprocessing.set_start_method('forkserver', force=True) # critical for make multiprocessing work
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal, Categorical
import gym
gym.logger.set_level(40) # suppress warnings (please remove if gives error)
import numpy as np
from collections import deque

torch.manual_seed(0) # set random seed
#from pyvirtualdisplay import Display
#display = Display(visible=0, size=(1400, 900))
#display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

#cross check with our solutions once you finish
from agents import BJAgent

MODEL_PATH = 'models/'


'''
Helper function from Udacity Deep Reinforcement Learning Nanodegree
#https://github.com/udacity/deep-reinforcement-learning/blob/master/monte-carlo/plot_utils.py
Many great examples with MIT License.
'''
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
'''
def plot_blackjack_values(V):

    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in V:
            return V[x,y,usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y,usable_ace) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()

def plot_policy(policy):

    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in policy:
            return policy[x,y,usable_ace]
        else:
            return 1

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(10, 0, -1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x,y,usable_ace) for x in x_range] for y in y_range])
        surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2', 2), vmin=0, vmax=1, extent=[10.5, 21.5, 0.5, 10.5])
        plt.xticks(x_range)
        plt.yticks(y_range)
        plt.gca().invert_yaxis()
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.grid(color='w', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0,1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)','1 (HIT)'])
            
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()
'''
'''
print(env.observation_space)
env.reset()
print(env.action_space)

print(f'State: {env.reset()}')
print(f'Result: {env.step(0)} Dealer has {sum(env.dealer)}') #next state, reward, done, info

state = env.reset()
print(f'Starting hands: You have {state[0]}. Dealer is showing {state[1]}')
done = False
while not done:
    action = int(input('Choose action: '))
    state, reward, done, info = env.step(action)
    print(f'After taking action {action}, You have {state[0]}. Usable ace: {state[2]}')
    if done:
        print(f'You have {state[0]}, Dealer has {sum(env.dealer)}, Reward: {reward}')
'''
################################# EVERYTHING ABOVE IS FOR BLACK JACK EXAMPLE ######################################

import numpy as np
from collections import defaultdict
import sys
import random
from tqdm import trange



def run_single_episode(policy=None):
    result = []
    state = env.reset()
    while True:
        action = policy[state] if policy else env.action_space.sample()
        next_state,reward,done,info = env.step(action)
        result.append((state,action,reward,next_state,done))
        state = next_state
        if done: break
    return(result) #must return a list of tuples (state,action,reward,next_state,done)

'''
#load saved optimal q dictionary
optimal_q = pickle.load(open(f'{MODEL_PATH}bj_optimal_q.pkl','rb'))
#convert to optimal policy
optimal_policy = {state:np.argmax(value) for state, value in optimal_q.items()}
#stick only as control 
stick_policy = {state: 0 for state,value in optimal_policy.items()}

n_trial = 100000
sticks = []
randoms = []
optimals = []
for i in trange(n_trial):
    _,_,rewards_stick,_,_ = zip(*run_single_episode(stick_policy))
    _,_,rewards_random,_,_ = zip(*run_single_episode())
    _,_,rewards_optimal,_,_ = zip(*run_single_episode(optimal_policy))
    sticks.append(rewards_stick[-1])
    randoms.append(rewards_random[-1])
    optimals.append(rewards_optimal[-1])
'''


class Policy(nn.Module):
    '''
    def __init__(self, env, gamma = 1.0, start_epsilon = 1.0, end_epsilon = 0.05, epsilon_decay = 0.99999):
        self.env = env
        self.n_action = self.env.action_space.n
        #self.n_state = self.
        self.policy = defaultdict(lambda: 0) #always stay as best policy
        self.v = defaultdict(lambda:0) #state value initiated as 0
        self.gamma = gamma
        
        #action values
        self.q = defaultdict(lambda: np.zeros(self.n_action)) #action value
        self.g = defaultdict(lambda: np.zeros(self.n_action)) #sum of expected rewards
        self.n_q = defaultdict(lambda: np.zeros(self.n_action)) #number of actions performed
        
        #epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay
    '''
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state, epsilon, env):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        print(probs[0].detach().numpy())
        action = np.random.choice(np.arange(4), p=probs[0].detach().numpy())
        if random.random() <= epsilon:
            action = env.action_space.sample()
        print("here are some actions:", action)
        m = Categorical(probs)
        #print(m.sample())
        action2 = m.sample() #just for logging dont need atm
        return action, m.log_prob(action2)
#     #get epsilon

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
#env = gym.make('Blackjack-v0')
env = gym.make('LunarLander-v2')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
hidden_size = 100
#env = NormalizedActions(gym.make('LunarLander-v2').unwrapped)

#env = gym.make('CartPole-v0')
#env = NormalizedActions(gym.make('CartPole-v0').unwrapped)
#env.seed(0)

print('observation space:', env.observation_space.shape[0])
print('action space:', env.action_space.n)
policy = Policy(n_states, hidden_size, n_actions).to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.01)

def reinforce(n_episodes=5000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
       
        rewards = []
        state = env.reset()
        for t in range(max_t):
            # keeping track of the prob weights of each trajectory and corresponding rewards
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
            
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        # calculate rewards
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        # calculate gradient
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
        
    return scores

start_epsilon = 1.0
end_epsilon = 0.05
epsilon_decay = 0.99999

def get_epsilon(n_episode):
    epsilon = max(start_epsilon * (epsilon_decay ** n_episode), end_epsilon)
    return(epsilon)

def run_episode(epsilon):
    result = []
    state = env.reset()
    print("here is the state:", state)
    while True:
        action, log_action = policy.act(state, epsilon, env)
        next_state,reward,done,info = env.step(action)
        result.append((state,action,reward))
        state = next_state
        #print(state)
        if done: 
            break
    return(result)
        
def reinforce_mc(env, num_episodes, gamma=1.0):
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    print(returns_sum)
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episode
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        e = get_epsilon(i_episode)
        transitions = run_episode(e)
        states,actions,reward = zip(*transitions)
        #print(e)
        print(states)
        traversed = []
        #print(states)
        #print(actions)
        saved_log_probs = []
        rewards = []
        state = env.reset()
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        #episode = []
        #state = env.reset()
        # generate episode, get params and initialize discounting
        '''
        while True:
            probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
            action = np.random.choice(np.arange(2), p=probs)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        '''
        #actions, log_prob = policy.act(state)
        #saved_log_probs.append(log_prob)
        #states, reward, done, _ = env.step(actions)
        #rewards.append(reward)
        # zip(*) unpacks the iterable episode
        #states, actions, rewards = zip(*episode)
        discounts = np.array([gamma**i for i in range(len(reward)+1)])
        print("discounts:", discounts)
        print(reward)
        #print(states)
        # update sum of returns, num of visits and action-value
        # estimate for each state-action pair in the episode
        for idx, state in enumerate(states):
            #print(state.type())
            #print(np.vstack(state).astype(np.float).ravel())
            state = tuple(list(state))
            #print(state)
            #state = map(tuple, state)
            #state = tuple(array_of_tuples)
            #print(actions[idx])
            #print(state)
            #print(returns_sum[state][actions[idx]])
            #print(sum(rewards[idx:] * discounts[:-idx-1]))
            returns_sum[state][actions[idx]] += sum(reward[idx:] * discounts[:-idx-1])
            N[state][actions[idx]] += 1.0
            Q[state][actions[idx]] = returns_sum[state][actions[idx]] / N[state][actions[idx]]
    return Q
    
scores = reinforce_mc(env, 500000)
V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) for k, v in Q.items())


#print('observation space:', env.observation_space)
#print('action space:', env.action_space)


    '''
    def get_epsilon(self,n_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay ** n_episode), self.end_epsilon)
        return(epsilon)
    
#     #select action based on epsilon greedy
    def select_action(self,state,epsilon):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        #print(self.policy[state])
        best_action = np.argmax(self.policy[state]) if state in self.q else self.env.action_space.sample()
        if random.random() > epsilon:
            action = best_action
        else:
            action = self.env.action_space.sample()
        return(action)
    
#     #run episode with current policy
    def run_episode(self, epsilon):
        result = []
        state = self.env.reset()
        while True:
            action = self.select_action(state,epsilon)
            next_state,reward,done,info = self.env.step(action)
            result.append((state,action,reward,next_state,done))
            state = next_state
            if done: 
                break
        return(result)
    
#     #update policy to reflect q values
    def update_policy_q(self):
        for state, value in self.q.items():
            self.policy[state] = np.argmax(value)
        
#     #mc control
    def mc_control_q(self,n_episode=500000,first_visit=True,update_every=1):
        for t in trange(n_episode):
            traversed = []
            e = self.get_epsilon(t)
            transitions = self.run_episode(e)
            states,actions,rewards,next_states,dones = zip(*transitions)
            #states = torch.FloatTensor(states).unsqueeze(0).to(device)
            #print(tuple(states))

            #mc prediction
            for i in range(len(transitions)):
                discounts = np.array([self.gamma**j for j in range(len(transitions)+1)])
                if first_visit and ((states[i][0],actions[i]) not in traversed):
                    traversed.append((states[i][0],actions[i]))
                    self.n_q[states[i][0]][actions[i]]+=1
                    self.g[states[i][0]][actions[i]]+= sum(rewards[i:]*discounts[:-(1+i)])
                    self.q[states[i][0]][actions[i]] = self.g[states[i][0]][actions[i]] / self.n_q[states[i][0]][actions[i]]
                else:
                    self.n_q[states[i][0]][actions[i]]+=1
                    self.g[states[i][0]][actions[i]]+= sum(rewards[i:]*discounts[:-(1+i)])
                    self.q[states[i][0]][actions[i]] = self.g[states[i][0]][actions[i]] / self.n_q[states[i][0]][actions[i]]
            #update policy every few episodes seem to be more stable
            if t % int(update_every * n_episode - 1) ==0:
                self.update_policy_q()
        #final policy update at the end
        self.update_policy_q()
    
    #mc control glie
    def mc_control_glie(self,n_episode=500000,lr=0.,update_every=1):
        for t in trange(n_episode):
            traversed = []
            e = self.get_epsilon(t)
            transitions = self.run_episode(e)
            states,actions,rewards,next_states,dones = zip(*transitions)
            
            #mc prediction
            for i in range(len(transitions)):
                discounts = np.array([self.gamma**j for j in range(len(transitions)+1)])
                traversed.append((states[i],actions[i]))
                self.n_q[states[i]][actions[i]]+=1
                g = sum(rewards[i:]*discounts[:-(1+i)])
                alpha = lr if lr > 0 else (1/self.n_q[states[i]][actions[i]])
                self.q[states[i]][actions[i]]+= alpha * (g - self.q[states[i]][actions[i]])
       
            #update policy every few episodes seem to be more stable
            if t % int(update_every * n_episode - 1)==0:
                self.update_policy_q()
        #final policy update at the end
        self.update_policy_q()
    
    #get state value from action value
    def q_to_v(self):
        for state, value in self.q.items():
            self.v[state] = np.max(value)
    '''