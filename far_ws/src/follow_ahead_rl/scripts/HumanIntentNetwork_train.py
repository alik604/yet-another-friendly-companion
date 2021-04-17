""" if you change `init_pos_robot` in `gym_gazeboros_ac.py`, then do not have the code `if done:    break`.
 As it will be an instant done. This is why I have `while max_itr > 0`
 rather then it being an infinitly running loop with being `done` as the exit condition"""

from time import sleep
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import gym_gazeboros_ac

from HumanIntentNetwork import HumanIntentNetwork

TRAIN_ON_ONLY_NEW = True

ENV_NAME = 'gazeborosAC-v0'
RANDOMSEED = 42

EPISODES = 100  # 1000     # Simulations
STEPS_PER_EPI = 45
EPOCHS = 400 # 1000     # Training
BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # env = gym.make(ENV_NAME).unwrapped
    # env.set_agent(0)
    # env.seed(RANDOMSEED)
    # torch.manual_seed(RANDOMSEED)
    # np.random.seed(RANDOMSEED)

    state_dim = 43#env.observation_space.shape[0]
    action_dim = 2#env.action_space.shape[0]
    # print(f'State_dim:  {state_dim}')
    # print(f'Action_dim: {action_dim}')

    save_local_1 = './model_weights/HumanIntentNetwork/list_of_human_state.csv'
    save_local_2 = './model_weights/HumanIntentNetwork/list_of_human_state_next.csv'

    # if TRAIN_ON_ONLY_NEW:
    #     list_of_human_state = []
    #     list_of_human_state_next = []
    # else:
    #     list_of_human_state = pd.read_csv(save_local_1).values.tolist()
    #     list_of_human_state_next = pd.read_csv(save_local_2).values.tolist()

    # mode = 0
    # for i in range(EPISODES):
    #     env.set_person_mode(mode % 5)
    #     mode += 1
    #     state = env.reset()

    #     max_itr = STEPS_PER_EPI
    #     while max_itr > 0:
    #         max_itr -= 1

    #         action = [0,0]
    #         state, reward, done, _ = env.step(action)

    #         human_state = list(state)
    #         list_of_human_state.append(human_state)
    #         # print(f"person pose: {human_state}")

    #         sleep(0.1)

    #         state, reward, done, _ = env.step(action)

    #         xy = env.get_person_pos()
    #         next_state = [xy[0], xy[1], state[2]]
    #         list_of_human_state_next.append(next_state)
    #         # print(f"Next human state: {next_state}")

    # env.close()

    # # print(f'Before: {len(list_of_human_state)} | {len(list_of_human_state_next)}')

    # if TRAIN_ON_ONLY_NEW:
    #     # deep copy and have parallel
    #     COPY_list_of_human_state_ = list_of_human_state.copy()
    #     COPY_list_of_human_state_next_ = list_of_human_state_next.copy()
        
    #     # TODO: WTF: Check existence before reading indiscriminately
    #     # extend copy with saved data
    #     # COPY_list_of_human_state_.extend(pd.read_csv(save_local_1).values.tolist())
    #     # COPY_list_of_human_state_next_.extend(pd.read_csv(save_local_2).values.tolist())

    #     # save data
    #     _ = pd.DataFrame(COPY_list_of_human_state_).to_csv(
    #         save_local_1, header=False, index=False)
    #     _ = pd.DataFrame(COPY_list_of_human_state_next_).to_csv(
    #         save_local_2, header=False, index=False)
    # else:
    #     # save data
    #     _ = pd.DataFrame(list_of_human_state).to_csv(
    #         save_local_1, header=False, index=False)
    #     _ = pd.DataFrame(list_of_human_state_next).to_csv(
    #         save_local_2, header=False, index=False)

    # print(f'After: {len(list_of_human_state)} | {len(list_of_human_state_next)}')

    list_of_human_state = pd.read_csv(save_local_1).values.tolist()
    list_of_human_state_next = pd.read_csv(save_local_2).values.tolist()

    model = HumanIntentNetwork(inner=128, input_dim=state_dim, output_dim=3)
    model.load_checkpoint()
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam((model.parameters()), lr=1e-3)

    human_state_tensor = torch.Tensor(list_of_human_state).to(device)
    next_human_state_tensor = torch.Tensor(list_of_human_state_next).to(device)

    losses = []
    for epoch in range(EPOCHS):
        _sum = 0
        for i in range(0, human_state_tensor.size(0), BATCH_SIZE):
            # print(f'{i} to {i+BATCH_SIZE}')

            target = next_human_state_tensor[i: i+BATCH_SIZE]
            input_batch = human_state_tensor[i: i+BATCH_SIZE]
            pred = model.forward(input_batch)

            optimizer.zero_grad()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            _sum += loss.item()

            # print(f'pred {pred}')
            # print(f'target {target}')

        if epoch % 100 == 0:
            model.save_checkpoint()
        losses.append(_sum)
        print(f'Epoch {epoch} | Loss_sum {_sum:.4f}')

    model.save_checkpoint()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss of (possibly) Pre-Trained model')
    plt.savefig('image.png')
    plt.show()

    print("END")
