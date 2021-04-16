import random
import numpy as np
import pandas as pd
import os

from time import sleep

import gym
import gym_gazeboros_ac

# Constants
ENV_NAME = 'gazeborosAC-v0'
N_EPISODES = 1
STEPS_PER_EPISODE = 45
ADDON_PREV_DATA = True


if __name__ == '__main__':
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)

    save_local_1 = './model_weights/HumanIntentNetwork/list_of_human_state.csv'
    save_local_2 = './model_weights/HumanIntentNetwork/list_of_human_state_next.csv'

    list_of_human_state = []
    list_of_human_state_next = []

    mode = 0
    for i in range(N_EPISODES):
        env.set_person_mode(mode % 5)   # Cycle through different person modes
        mode += 1
        state = env.reset()

        max_itr = STEPS_PER_EPISODE
        while max_itr > 0:
            max_itr -= 1

            action = [0,0]
            state, reward, done, _ = env.step(action)

            human_state = list(state)
            list_of_human_state.append(human_state)

            sleep(0.1)

            state, reward, done, _ = env.step(action)

            xy = env.get_person_pos()
            next_state = [xy[0], xy[1], state[2]]
            list_of_human_state_next.append(next_state)

    env.close()

    if ADDON_PREV_DATA:
        if os.path.isfile(save_local_1) and os.path.isfile(save_local_2):
            list_of_human_state.extend(pd.read_csv(save_local_1).values.tolist())
            list_of_human_state_next.extend(pd.read_csv(save_local_2).values.tolist())
        else:
            print("Warning: Tried to load previous data but files were not found!")


    # save data
    _ = pd.DataFrame(list_of_human_state).to_csv(
        save_local_1, header=False, index=False)
    _ = pd.DataFrame(list_of_human_state_next).to_csv(
        save_local_2, header=False, index=False)

    print("END")
