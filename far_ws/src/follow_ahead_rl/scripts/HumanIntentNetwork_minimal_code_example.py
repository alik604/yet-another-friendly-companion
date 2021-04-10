import random
import numpy as np

import torch
import gym
import gym_gazeboros_ac

from HumanIntentNetwork import HumanIntentNetwork

ENV_NAME = 'gazeborosAC-v0'
RANDOMSEED = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    print('START Move Test')
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)
    env.seed(RANDOMSEED)
    torch.manual_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)

    state = env.reset()

    # Prints out x y position of person
    # print(f"person pose = {env.get_person_pos()}") # hooman state x,y,z
    # print(state)


    action = [random.uniform(-1, 1), random.uniform(-1, 1)]
    print(action)

    state, reward, done, _ = env.step(action)
    human_state = list(env.get_person_pos())
    print(f"person pose 1st: {human_state}")

    state, reward, done, _ = env.step(action)
    human_state_next = list(env.get_person_pos())
    print(f"person pose 2nd: {human_state_next}")

    # Prints out system velocities
    # print(f"system_velocities = {env.get_system_velocities()}")

    env.close()

    model = HumanIntentNetwork(inner=24)
    model.load_checkpoint()
    model.to(device)

    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam((model.parameters()), lr=1e-3)

    human_state = torch.Tensor([human_state, human_state]).to(device)
    human_state_next = torch.Tensor([human_state_next, human_state_next]).to(device)
    
    assert human_state.size() == human_state_next.size()
    print(human_state.size())
    print(human_state_next.size())

    for i in range(0, human_state.size(0)):

        target = human_state[i] 
        pred = model.forward(human_state_next[i])

        # optimizer.zero_grad()
        loss = criterion(pred, target)
        loss.backward()
        # optimizer.step()
        print(f'\n\nPrediction of {target.cpu().detach().numpy()} is {pred.cpu().detach().numpy()} | loss {loss.item():.4f}')



    print("\nEND")
