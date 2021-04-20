
import math
import gym
import time
import matplotlib.pyplot as plt

import gym_gazeboros_ac


ENV_NAME = 'gazeborosAC-v0'
NUM_EPISODES = 100
EPISODE_LEN = 15
EVALUATE_PIPELINE = True

class DistanceHeuristic:
    # Args:
    # target_distance: desired distance in front of target
    def __init__(self, target_distance=0.45):
        self.target_distance = target_distance
    
    def rotate_vector(self, xy, orientation):
        x = math.cos(orientation) * xy[0] - math.sin(orientation) * xy[1]
        y = math.sin(orientation) * xy[0] + math.cos(orientation) * xy[1]

        return [x,y]

    # Args:
    # target_predicted_state: [x,y,theta]
    def calculate_goal(self, target_predicted_state):
        vector = [self.target_distance*4, 0]
        vector = self.rotate_vector(vector, target_predicted_state[2])

        goal = [target_predicted_state[0] + vector[0], target_predicted_state[1] + vector[1]]
        return goal

    # Args:
    # target_predicted_state: [x,y,theta]
    def calculate_vector(self, target_predicted_state):
        vector = [self.target_distance, 0]

        return vector


if __name__ == '__main__':
    
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)
    
    dis_heuristic = DistanceHeuristic()

    cumulative_reward_per_episode = []
    cumulative_reward = 0
    mode = 0
    for i in range(NUM_EPISODES):
        if EVALUATE_PIPELINE:
            if (mode % 5) == 3:     # Don't use random person path
                mode += 1
        env.set_person_mode(mode % 5)
        print(f"Running Episode {i} Person Mode = {mode % 5}")
        mode += 1
        state = env.reset()

        # Currently set to run for 15 seconds
        for i in range(EPISODE_LEN * 5):
            person_state = env.get_person_pos()

            action = dis_heuristic.calculate_vector(person_state)
            
            goal = dis_heuristic.calculate_goal(person_state)            
            env.set_marker_pose(goal)

            state, reward, done, _ = env.step(action)
            cumulative_reward += reward
            time.sleep(0.1)
            
        cumulative_reward_per_episode.append(cumulative_reward)
    
    print(f"Cumulative reward: {cumulative_reward_per_episode}")
    
    plt.plot(cumulative_reward_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Distance Heuristic Evaluation Cumulative Reward')

    plt.savefig('DH_EVAL_CR.png')
    plt.show()
    
    print("END")