from gym.envs.registration import register

register(
    id='gazeborosAC-v0',
    entry_point='gym_gazeboros_ac.envs.gym_gazeboros_ac:GazeborosEnv',
)
