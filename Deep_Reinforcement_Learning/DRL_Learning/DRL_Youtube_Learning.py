import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

'''

## Randomly trying to choose between going left and right

environment_name = 'CartPole-v0'
env = gym.make(environment_name)

episodes = 5
for episodes in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, _, info = env.step(action)
        score += reward
    print('Episode : {} Score : {}'.format(episodes,score))

env.close()
'''
##########################################################################################################################################
'''

### Training RL model to make the program understand either to go left or right


log_path = os.path.join('Training','Logs')
#log_path = FILE_DIR_PATH/'Logs'
environment_name = 'CartPole-v0'
env = gym.make(environment_name)
env = DummyVecEnv([lambda:env])

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000)

PPO_Path = os.path.join('Training','Saved Models', 'PPO_Model_Cartpole')
model.save(PPO_Path) ### Trained model here and deleting it at the next line

del model

environment_name = 'CartPole-v0'
env = gym.make(environment_name)
PPO_Path = os.path.join('Training','Saved Models', 'PPO_Model_Cartpole')
model = PPO.load(PPO_Path,env) ###  Loading the model again from the saved file
print(evaluate_policy(model,env,n_eval_episodes=10,render=True))


episodes = 5
for episodes in range(1, episodes+1):
    obs, _ = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        score += reward
    print('Episode : {} Score : {}'.format(episodes,score))

env.close()

'''

##########################################################################################################################################

'''
### Training RL model to make the program understand either to go left or right and then saving the most perfect one

environment_name = 'CartPole-v0'
env = gym.make(environment_name)
log_path = os.path.join('Training','Logs')
save_path = os.path.join('Training', 'Saved Models')
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(env, ########## Saving the best possible model which has a score of more than 200 . Fot this environment 200 is seen as a good score. ep_len_mean  is the score
                             callback_on_new_best=stop_callback,
                             eval_freq=100000,
                             best_model_save_path=save_path,
                             verbose=1)


model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000,callback=eval_callback)
'''


############################################################################################################################

'''

### Updating the neural networks in our model to update the original neural network of the PPO algorithm. No need to that as
### original PPO algorithm neural networks already works pretty good.

environment_name = 'CartPole-v0'
env = gym.make(environment_name)
log_path = os.path.join('Training','Logs')
net_arch = [dict(pi=[128,128,128,128], vf = [128,128,128,128])] ### Defining my own neural network for the PPO
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch' : net_arch})
model.learn(total_timesteps= 20000)

'''

##############################################################################################################################


'''
## Just using a different algorithm that is DQN instead of PPO

from stable_baselines3 import DQN

environment_name = 'CartPole-v0'
env = gym.make(environment_name)
log_path = os.path.join('Training','Logs')
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000)

DQN_Path = os.path.join('Training','Saved Models', 'DQN_Model_Cartpole')
model.save(DQN_Path)

del model

DQN_Path = os.path.join('Training','Saved Models', 'DQN_Model_Cartpole')
model = PPO.load(DQN_Path,env)

'''





