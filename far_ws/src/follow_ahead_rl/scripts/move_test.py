import gym
import gym_gazeboros_ac

from time import sleep

ENV_NAME = 'gazeborosAC-v0'
EPISODE_LEN = 20

# Robot Chase Simulator 2021
# How to use:
# Terminal 1: Launch turtlebot.launch
# Terminal 2: run `python tf_node.py in old_scripts`
# Terminal 3: Launch navigation.launch
# Terminal 4: run this file
#
# * DON'T FORGET TO SOURCE THE WORKSPACE IN EACH FILE <3
# ie: cd .../far_ws && source devel/setup.bash

if __name__ == '__main__':
    print('START Move Test')
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)


    while True:
        state = env.reset()
        c = 0
        for i in range(EPISODE_LEN):


            action = [0, 0]
            state, reward, done, _ = env.step(action)
            
            sleep(1)

            # if done:
            #     break

            c += 1
    
    print("END")