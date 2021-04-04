
'''
NN does:
 - Take human intent (anthony mades function)
 - - loss. MSE(human.next_state and human.next_state_prime)
 - Output possible moves of human

 - Pass into MCTS
 - MCST outputs best action

Does this sound like Double Q learning?
--------------------------------------------------------------
How to use:
Terminal 1: Launch turtlebot.launch
Terminal 2: run `python tf_node.py in old_scripts`
Terminal 3: Launch navigation.launch
Terminal 4: run this file

* DON'T FORGET TO SOURCE THE WORKSPACE IN EACh TERMINAL
ie: cd .../far_ws && source devel/setup.bash

if you have a issue with `tf_node.py`, follow this https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/

'''

from time import sleep
import random
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
import gym
import gym_gazeboros_ac
import pandas as pd


TRAIN_ON_ONLY_NEW = False

ENV_NAME = 'gazeborosAC-v0'
RANDOMSEED = 42

EPISODES = 1000     # Simulations
EPOCHS = 2000       # Training 
BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RbfNet(nn.Module):
    '''
    theory:    http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
    code:      https://github.com/csnstat/rbfn
    code-alt:  https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer
    '''

    def __init__(self, centers=500, num_class=2):
        super(RbfNet, self).__init__()
        self.num_class = num_class
        self.num_centers = centers  # .size(0)

        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1, self.num_centers)/10)
        self.linear = nn.Linear(self.num_centers, self.num_class, bias=True)
        utils.initialize_weights(self)

    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1,
                                                          self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A-B).pow(2).sum(2, keepdim=False).sqrt()))
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(radial_val)
        return class_score

class HumanIntentRegressor(nn.Module):
    def __init__(self, inner=24, chkpt_dir='./model_weights/HumanIntentRegressor'):
        super(HumanIntentRegressor, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 'HumanIntentRegressor_deeper')
        self.fc1 = nn.Linear(2, inner)
        self.fc2 = nn.Linear(inner, inner*2)
        self.fc2_5 = nn.Linear(inner*2, inner)
        self.fc3 = nn.Linear(inner, 2)
        self.selu = nn.SELU() # # ReLU, LeakyReLU

    def forward(self, x):
        x = self.selu(self.fc1(x))  
        x = self.selu(self.fc2(x))
        x = self.selu(self.fc2_5(x))
        x = self.fc3(x)
        return x

    def save_checkpoint(self):
        print('\n... saving checkpoint ...\n')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))



# inner=24
# model = nn.Sequential(
#         nn.Linear(2, inner),
#         nn.SELU(),
#         nn.Linear(inner, inner),
#         nn.SELU(),
#         nn.Linear(inner, 2),
#     )


if __name__ == '__main__':
    print('START Move Test')
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)
    env.seed(RANDOMSEED)
    torch.manual_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f'State_dim:  {state_dim}')
    print(f'Action_dim: {action_dim}')


    save_local_1 = './model_weights/HumanIntentRegressor/list_of_human_state.csv'
    save_local_2 = './model_weights/HumanIntentRegressor/list_of_human_state_next.csv'

    if TRAIN_ON_ONLY_NEW: 
        list_of_human_state = []
        list_of_human_state_next = [] 
    else: 
        list_of_human_state = pd.read_csv(save_local_1).values.tolist()
        list_of_human_state_next = pd.read_csv(save_local_2).values.tolist()

    for i in range(EPISODES):
        # env.set_obstacle_pos("obstacle_box",0.5, 0, 0)
        state=env.reset()

        # Prints out x y position of person
        # print(f"person pose = {env.get_person_pos()}") # hooman state x,y,z
        # print(state)

        while True:
            action=[random.uniform(-1, 1), random.uniform(-1, 1)]
            print(action)

            state, reward, done, _ = env.step(action)
            if done:    break
            human_state=list(env.get_person_pos())
            list_of_human_state.append(human_state)
            # print(f"person pose: {human_state}")

            state, reward, done, _ = env.step(action)
            if done:    break # will need to discard non-parallel end 
            human_state_next=list(env.get_person_pos())
            list_of_human_state_next.append(human_state_next)
            # print(f"person pose: {human_state_next}")

            # Prints out system velocities
            # print(f"system_velocities = {env.get_system_velocities()}")

        if i % 1 == 0: # i > 0 and
            print(f'\n\n\n It has been {i} Simulations \n\n\n')
    env.close()


    # discard non-parallel end 
    cuttoff = min(len(list_of_human_state), len(list_of_human_state_next))
    list_of_human_state = list_of_human_state[:cuttoff]
    list_of_human_state_next = list_of_human_state_next[:cuttoff]
    
    if TRAIN_ON_ONLY_NEW: 
        # deep copy and have parallel
        cuttoff = len(list_of_human_state_next)
        list_of_human_state_ = list_of_human_state.copy()
        list_of_human_state_next_ = list_of_human_state_next.copy()

        # extend with saved data
        list_of_human_state_.extend(pd.read_csv(save_local_1).values.tolist())
        list_of_human_state_next_.extend(pd.read_csv(save_local_2).values.tolist())

        # save data
        _ = pd.DataFrame(list_of_human_state_).to_csv(save_local_1, header=False, index=False)
        _ = pd.DataFrame(list_of_human_state_next_).to_csv(save_local_2, header=False, index=False)
    else:
        # save data
        _ = pd.DataFrame(list_of_human_state).to_csv(save_local_1, header=False, index=False)
        _ = pd.DataFrame(list_of_human_state_next).to_csv(save_local_2, header=False, index=False)

    
    print(len(list_of_human_state))
    print(len(list_of_human_state_next))

    model = HumanIntentRegressor(inner=24)
    model.load_checkpoint()
    model.to(device) 

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam((model.parameters()), lr=1e-3)

    # TODO shuffle data, in a parallel manner 

    list_of_human_state = torch.Tensor(list_of_human_state).to(device)
    list_of_human_state_next = torch.Tensor(list_of_human_state_next).to(device)
    print(list_of_human_state.size(0))

    losses = []
    for epoch in range(EPOCHS): 
        summer = 0
        for i in range(0, list_of_human_state.size(0), BATCH_SIZE):
            # print(f'{i} to {i+BATCH_SIZE}')

            target = list_of_human_state_next[i: i+BATCH_SIZE]
            input_batch = list_of_human_state[i: i+BATCH_SIZE]
            pred = model.forward(input_batch)

            optimizer.zero_grad()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()    
            summer += loss.item()

        if epoch % 100:
            model.save_checkpoint()
        losses.append(summer)
        print(f'Epoch {epoch} | Loss_sum {summer:.4f}')

    model.save_checkpoint()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss of (possibly) Pre-Trained model')
    plt.show()

    print("END")
