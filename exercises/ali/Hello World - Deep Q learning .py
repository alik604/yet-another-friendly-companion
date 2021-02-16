import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import math
import numpy as np
import random

class Network(torch.nn.Module):
    def __init__(self, alpha, inputShape, numActions):
        super().__init__()
        self.inputShape = inputShape
        self.numActions = numActions
        self.fc1Dims = 1024
        self.fc2Dims = 512

        self.fc1 = nn.Linear(*self.inputShape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, numActions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Agent():
    def __init__(self, lr, inputShape, numActions):
        self.network = Network(lr, inputShape, numActions)

    def chooseAction(self, observation):
        state = torch.tensor(observation).float().detach()
        state = state.to(self.network.device)
        state = state.unsqueeze(0)

        qValues = self.network(state)
        action = torch.argmax(qValues).item()

        chanceOfAsparagus = random.randint(1, 10)
        if chanceOfAsparagus == 1:  #   10% chance
            action = random.randint(0, 1)

        # print("qValues: {}, action {}".format(qValues.detach(), action))
        return action

    def learn(self, state, action, reward, state_, done):
        self.network.optimizer.zero_grad()

        state = torch.tensor(state).float().detach().to(self.network.device).unsqueeze(0)
        state_ = torch.tensor(state_).float().detach().to(self.network.device).unsqueeze(0)
        reward = torch.tensor(reward).float().detach().to(self.network.device)

        qValues = self.network(state)
        nextQValues = self.network(state_)

        predictedValueOfNow = qValues[0][action]    #   interpret the past
        futureActionValue = nextQValues[0].max()    #   interpret the future

        trueValueOfNow = reward + futureActionValue * (1 - done)

        loss = self.network.loss(trueValueOfNow, predictedValueOfNow)

        loss.backward()
        self.network.optimizer.step()

if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(lr=0.0001, inputShape=(4,), numActions=2)

    highScore = -math.inf
    episode = 0
    for i in range(5000):
        done = False
        state = env.reset()

        score, frame = 0, 1
        while not done:
            env.render()

            action = agent.chooseAction(state)
            state_, reward, done, info = env.step(action)

            agent.learn(state, action, reward, state_, done)
            state = state_

            score += reward
            frame += 1
            # print("reward {}".format(reward))

        highScore = max(highScore, score)

        print(( "ep {}: high-score {:12.3f}, "
                "score {:12.3f}, last-episode-time {:4d}").format(
            episode, highScore, score,frame))

        episode += 1
