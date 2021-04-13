
import math
import gym
from time import sleep

import gym_gazeboros_ac


ENV_NAME = 'gazeborosAC-v0'
EPISODE_LEN = 15


class DistanceHeuristic:
    # Args:
    # target_distance: desired distance in front of target
    def __init__(self, target_distance=0.5):
        self.target_distance = target_distance
    
    def rotate_vector(self, xy, orientation):
        x = math.cos(orientation) * xy[0] - math.sin(orientation) * xy[1]
        y = math.sin(orientation) * xy[0] + math.cos(orientation) * xy[1]

        return [x,y]

    # Args:
    # target_predicted_state: [x,y,theta]
    # obstacle_states: [(xy and size)]
    def calculate_goal(self, target_predicted_state, obstacle_states):
        vector = [self.target_distance, 0]
        # vector = self.rotate_vector(vector, target_predicted_state[2])
        vector = self.rotate_vector(vector, math.pi/2)

        # TODO: Deal with obstacles

        goal = [target_predicted_state[0] + vector[0], target_predicted_state[1] + vector[1]]

        return goal
    
    # Args:
    # target_predicted_state: [x,y,theta]
    # obstacle_states: [(xy and size)]
    def calculate_vector(self, target_predicted_state, obstacle_states):
        vector = [self.target_distance, 0]
        # vector = self.rotate_vector(vector, target_predicted_state[2])

        # TODO: Deal with obstacles

        return vector


if __name__ == '__main__':
    
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)
    
    dis_heuristic = DistanceHeuristic(target_distance=0.4)

    mode = 0
    while True:
        env.set_person_mode(mode % 5)
        mode += 1
        state = env.reset()


        for i in range(EPISODE_LEN * 10):
            person_state = env.get_person_pos()

            action = dis_heuristic.calculate_vector(person_state, [])

            state, reward, done, _ = env.step(action)
            # print(state)

            sleep(0.1)

    
    print("END")