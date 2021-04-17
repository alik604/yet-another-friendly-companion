from time import sleep
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import gym_gazeboros_ac

from HumanIntentNetwork import HumanIntentNetwork

ENV_NAME = 'gazeborosAC-v0'
EPOCHS = 400
BATCH_SIZE = 64


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dim = 43
    action_dim = 2

    save_local_1 = './model_weights/HumanIntentNetwork/list_of_human_state.csv'
    save_local_2 = './model_weights/HumanIntentNetwork/list_of_human_state_next.csv'

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
