'''
NN does:
 - Takes state 
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

import os
import torch
import torch.nn as nn
import torch.optim as optim


class RBF_HumanIntentNetwork(nn.Module):
    '''
    theory:    http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
    code:      https://github.com/csnstat/rbfn
    code-alt:  https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer
    '''

    def __init__(self, centers=500, num_class=2):
        super(RBF_HumanIntentNetwork, self).__init__()
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


class HumanIntentNetwork(nn.Module):
    def __init__(self, inner=128, input_dim=23, output_dim=3, chkpt_dir='./model_weights/HumanIntentNetwork'):
        super(HumanIntentNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, 'HumanIntentNetwork')
        self.fc1 = nn.Linear(input_dim, inner) 
        self.fc2 = nn.Linear(inner, inner*4)
        self.fc2_5 = nn.Linear(inner*4, inner)
        self.fc3 = nn.Linear(inner, output_dim)
        self.selu = nn.SELU()  # ReLU, LeakyReLU

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
        try:
            self.load_state_dict(torch.load(self.checkpoint_file))
        except Exception as e:
            print('... FAILED: loading checkpoint ...')
