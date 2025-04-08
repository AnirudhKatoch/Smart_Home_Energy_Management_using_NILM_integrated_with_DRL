from typing import Optional, Union, List
from gym import Env
from gym.core import RenderFrame
from gym.spaces import Discrete, Box
import numpy as np
import random

class ShowerEnv(Env):

    def __init__(self):

        # Actions we can take - down, stay, up
        self.action_space = Discrete(3)

        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))

        # set start temperature
        self.state = 38 + random.randint(-3,3)

        # set shower length
        self.shower_length = 60


    def step(self, action):

        # Apply Action
        self.state += action - 1

        # Reduce Shower Length by 1 second
        self.shower_length -= 1

        # Calculate Reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

        # Check if shower is done
        if self.shower_length <=0:
            done = True
        else:
            done = False

        # Apply temperature noise
        self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info


    def render(self):

        pass

    def reset(self):

        # Reset start temperature
        self.state = 38 + random.randint(-3,3)

        # Reset shower length
        self.shower_length = 60

        return  self.state

env = ShowerEnv()

#print(env.action_space.sample())
#print(env.observation_space.sample())


episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward

    print('Episode : {}, Score : {}'.format(episode, score))

env.close()