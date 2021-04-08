# 1. selection: starts at rood node R then moves down the tree  by selecting optimal child nodes until a leaf node L is reached
# 2. Expansion: if L is not a terminal node then create one or more child nodes according to available actions at current state,
#  then select the first of these new nodes
# 3. Simulation: Run a simulated rollout from M until terminal state is found
# 4. backpropogation
# Output: goal (x, y position) or velocity -- goals as edges, pick reward function

import random
import sys
import warnings
import pickle

from tqdm import trange
from collections import deque, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gym
import torch
# import torch as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal, Categorical

# critical for make multiprocessing work
torch.multiprocessing.set_start_method('forkserver', force=True)
gym.logger.set_level(40)  # suppress warnings (please remove if gives error)

torch.manual_seed(0)  # set random seed
#from pyvirtualdisplay import Display
#display = Display(visible=0, size=(1400, 900))
# display.start()

device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# cross check with our solutions once you finish
#from agents import BJAgent

MODEL_PATH = 'model_weights/'


'''
Helper function from Udacity Deep Reinforcement Learning Nanodegree
#https://github.com/udacity/deep-reinforcement-learning/blob/master/monte-carlo/plot_utils.py
Many great examples with MIT License.
'''

'''
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


class MCTS(): # nn.Module
    def __init__(self, env, obs_size, hidden_size, n_actions, gamma=1.0, start_epsilon=1.0, end_epsilon=0.05, epsilon_decay=0.99991):  # 0.99999
        super(MCTS, self).__init__()
        self.env = env
        self.n_action = self.env.action_space.n
        # self.n_state = self.
        self.p = defaultdict(lambda: 0)  # always stay as best policy
        self.v = defaultdict(lambda: 0)  # state value initiated as 0
        self.gamma = gamma

        # action values
        self.Q = defaultdict(lambda: np.zeros(self.n_action))  # action value
        # sum of expected rewards
        self.N = defaultdict(lambda: np.zeros(self.n_action))
        self.returns_sum = defaultdict(lambda: np.zeros(
            self.n_action))  # number of actions performed

        # epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay

    # def forward(self, x):
    #    x = F.relu(self.fc1(x))
    #    x = self.fc2(x)
    #    return F.softmax(x, dim=1)

    # def select_action(self,state,epsilon):
    #    state = torch.FloatTensor(state).unsqueeze(0).to(device)
        # print(self.policy[state])
    #    best_action = np.argmax(self.policy[state]) if state in self.q else self.env.action_space.sample()
    #    if random.random() > epsilon:
    #        action = best_action
    #    else:
    #        action = self.env.action_space.sample()
    #    return(action)

    def act(self, state, epsilon, env):
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # probs = self.forward(state).cpu()
        # print(probs[0].detach().numpy())
        # action = np.random.choice(np.arange(4), p=probs[0].detach().numpy())

        # action = np.argmax(self.p[state]) if state in self.Q else self.env.action_space.sample()
        print(state)
        print(self.Q[state])
        print(state in self.Q)
        
        if state in self.Q:
            action = np.argmax(self.p[state])
            print('***state in self.Q')
        else: 
            action = self.env.action_space.sample()
        exit(1)   

        # print(f'epsilon is: {epsilon}')
        if random.random() <= epsilon:
            action = self.env.action_space.sample()
        elif random.random() <= 0.0001:  # Print 0.01% of the time. Have mercy on std::out and thank speculative exec
            print(f'Selected greedy action. epsilon is: {epsilon:.4f}')

        #print("here are some actions:", action)
        #m = Categorical(probs)
        # print(m.sample())
        # action2 = m.sample() #just for logging dont need atm
        return action

    def update_policy_q(self):
        '''
        Update policy `p` to reflect new `q` values
        '''
        for state, value in self.Q.items():
            self.p[state] = np.argmax(value)

    def get_epsilon(self, n_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay ** n_episode), self.end_epsilon)
        return(epsilon)

    def run_episode(self, epsilon):
        # result = []
        states = []
        actions = [] 
        rewards = [] 
        state = self.env.reset()
        # print("The state is:", state)
        while True:
            action = self.act(state, epsilon, env)
            next_state, reward, done, info = env.step(action)
            # result.append((state, action, reward))
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

            if done:
                break
        return states, actions, rewards # (result)

    def reinforce_mc(self, env, num_episodes, gamma=1.0, update_every=1):

        # initialize empty dictionaries of arrays
        # returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
        # N = defaultdict(lambda: np.zeros(env.action_space.n))
        # Q = defaultdict(lambda: np.zeros(env.action_space.n))
        # print(returns_sum)

        # loop over episode
        rewards = []
        for i_episode in range(num_episodes):  # range(1, num_episodes+1):
            # monitor progress
            epsilon = self.get_epsilon(i_episode)
            # transitions = self.run_episode(e)
            # states, actions, reward = zip(*transitions)

            states, actions, reward = self.run_episode(epsilon)
            # traversed = []
            print(reward)
            # print(actions)
            # saved_log_probs = []
            
            state = self.env.reset()
            if i_episode % 500 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
                sys.stdout.flush()
            discounts = np.array([gamma**i for i in range(len(reward)+1)])
            rewards.append(reward)
            # print(states)
            # update sum of returns, num of visits and action-value
            # estimate for each state-action pair in the episode
            for idx, state in enumerate(states):
                if type(state) != tuple:
                    state = tuple(list(state))
                self.returns_sum[state][actions[idx]
                                        ] += sum(reward[idx:] * discounts[:-idx-1])
                self.N[state][actions[idx]] += 1.0
                self.Q[state][actions[idx]] = self.returns_sum[state][actions[idx]] / self.N[state][actions[idx]]
            # Ali: `int(update_every * num_episodes - 1)`. It seemed to update only twice, together, at the end...
            if i_episode % int(update_every) == 0:
                # print(f'i_episode is {i_episode}')
                self.update_policy_q()

        # final policy update at the end
        self.update_policy_q()
        return self.Q, rewards

    def q_to_v(self):
        for state, value in self.Q.items():
            self.v[state] = np.max(value)


# env = gym.make('Blackjack-v0')
env = gym.make('LunarLander-v2')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
hidden_size = 100
num_episodes = 50 #40000

print('observation space:', env.observation_space.shape[0])
print('action space:', env.action_space.n)

policy = MCTS(env, n_states, hidden_size, n_actions) # .to(device)

epsilon_at_end = policy.get_epsilon(num_episodes)
# even if this condistion is not met, Policy may still be inadequate in exploitation
if epsilon_at_end > policy.end_epsilon: 
    print(f'Policy is inadequate in exploitation. epsilon_at_end: {epsilon_at_end:.2f} | policy.end_epsilon: {policy.end_epsilon}')
    # exit(0) 

scores, reward = policy.reinforce_mc(env, num_episodes, update_every=1) # 250

print("here are the number of rewards\n: ", len(reward[0]))
print("here are the rewards: ", np.round(np.asarray(reward[0]), 2))

timer = np.arange(int(len(scores[0])))

print(scores)

plt.plot(scores[0])
# plt.plot(np.arange(int(len(scores[-1]))), scores[-1])
plt.title('scores over episodes')
plt.xlabel('time')
plt.ylabel('scores')
plt.show()

print(f'SCORES: {type(scores)} | {type(scores[0])}')
# optimizer = optim.Adam(policy.parameters(), lr=0.01)


# scores, reward = reinforce_mc(env, 50)
# print(scores)


# V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) for k, v in Q.items())
# print('observation space:', env.observation_space)
# print('action space:', env.action_space)


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
