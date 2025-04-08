# For Environment
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

# For Reinforcement Learning
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

import numpy as np
import random
import os

#print(Discrete(3).sample())
#print(Box(0,1,shape=(3,3)).sample())
#print(Box(0,1,shape=(3,)).sample())
#print((Tuple((Discrete(3),Box(0,1,shape=(3,))))).sample())
#print((Dict({'height': Discrete(2), 'speed':Box(0,100,shape=(1,))})).sample())
#print(MultiBinary(4).sample())
#print(MultiDiscrete([5,2,2]).sample())

class ShowerEnv(Env):
    def __init__(self):

        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high = np.array(([100])))
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60


    def step(self,action):

        self.state += action-1
        self.shower_length -= 1

        if self.state >=37 and self.state<=39:
            reward = 1
        else:
            reward = -1

        if self.shower_length <=0:
            done = True
        else:
            done = False

        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):

        self.state = np.array([38 + random.randint(-3, 3)]).astype(float)
        self.shower_length = 60

        return self.state

env = ShowerEnv()

#print(env.observation_space.sample())
#print(env.action_space.sample())

'''
episodes = 5
for episodes in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode : {} Score : {}'.format(episodes,score))

env.close()
'''

log_path = os.path.join('Training', 'Logs')
model = PPO('MlpPolicy',env,verbose = 1, tensorboard_log = log_path)
model.learn(total_timesteps=40000)

'''
save_path = os.path.join('Training', 'Shower Saved Models')
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=10, verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=100000,
                             best_model_save_path=save_path,
                             verbose=1)


model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=200000,callback=eval_callback)
'''





#Shower_Path = os.path.join('Training','Saved Models', 'PPO_Shower_Model')
#model.save(Shower_Path)

#del model

#Shower_Path = os.path.join('Training','Shower Saved Models', 'best_model')
#model = PPO.load(Shower_Path,env)
#print(evaluate_policy(model,env,n_eval_episodes=10,render=True))