import gym
from gym import spaces
import numpy as np
from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os

FILE_DIR_PATH = Path(__file__).parent.parent

class LoadShiftingEnv(gym.Env):

    def __init__(self, pv_power, electricity_price, feed_in_tariff, appliance_table):

        super(LoadShiftingEnv, self).__init__()

        # Inputs: PV power, electricity price, and feed-in price (each has 96 steps)

        self.pv_power = pv_power
        self.electricity_price = electricity_price
        self.feed_in_tariff = feed_in_tariff
        self.total_steps = len(pv_power)

        self.Washing_machine_run_time = appliance_table['Washing_Machine'][0]
        self.Washing_machine_power = appliance_table['Washing_Machine'][1]

        # Action space: 90 possible time slots (when to start the washing machine)
        self.action_space = spaces.Discrete(self.total_steps - self.Washing_machine_run_time)

        # Observation space: [time step, pv power, electricity price, feed-in price]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)

        # Initialize the environment
        self.reset()

    def reset(self):

        self.current_step = 0
        self.washing_machine_start = 0
        self.best_action = None
        self.best_reward = -np.inf
        self.done = False

        return self._next_observation()

    def _next_observation(self):

        return np.array([
            self.current_step,
            self.pv_power[self.current_step],
            self.electricity_price[self.current_step],
            self.feed_in_tariff[self.current_step],

        ], dtype=np.float32)

    def step(self, action):

        if self.done:

            return self._next_observation(), 0, self.done, {}

        self.washing_machine_start = action + 1

        total_feed_in = 0
        total_supply_cost = 0

        for i in range(self.total_steps):

            if self.washing_machine_start <= i + 1 < self.washing_machine_start + self.Washing_machine_run_time:
                net_power = self.pv_power[i] - self.Washing_machine_power
            else:
                net_power = self.pv_power[i]

            if net_power > 0:

                total_feed_in += net_power * (1/60) * self.feed_in_tariff[i]

            elif net_power == 0:

                total_feed_in += 0
                total_supply_cost += 0

            else:

                total_supply_cost += abs(net_power) * (1/60) * self.electricity_price[i]

        total_reward = total_feed_in - total_supply_cost

        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_action = action

        self.done = True

        return self._next_observation(), total_reward, self.done, {}

    def get_best_action(self):
        return self.best_action, self.best_reward

    def render(self, mode='human'):
        pass

    def close(self):
        pass


# Example Data (replace with your actual data)
#pv_power = np.random.uniform(0, 2000, 96)  # Simulated PV power in W
#electricity_price = np.random.uniform(0.1, 0.3, 96)  # Simulated electricity price in $/Wh
#feed_in_tariff = np.random.uniform(0.05, 0.15, 96)  # Simulated feed-in tariff in $/Wh

df = pd.read_csv(FILE_DIR_PATH / 'Database/PV_elec/1_min_resolution/DRL_1_day_mean_inputs.csv', sep=';')

pv_power = np.array(df['PV_Power_W']) # PV Power (in W)
electricity_price = np.array(df['Electricity_Price_EUR_Wh'])  # Grid electricity price (in $/Wh)
feed_in_tariff = np.array(df['Feed_in_Tariff_EUR_Wh'])  # Feed-in tariff (in $/Wh)
appliance_table = pd.read_csv(FILE_DIR_PATH/'Database/Appliance_Specifications.csv',sep=';') # Appliance Table

# Create the environment
env = LoadShiftingEnv(pv_power, electricity_price, feed_in_tariff, appliance_table)

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=50000)

# Save the model
model.save("Models/load_shifting_ppo_Washing_Machine")

del model

model = PPO.load("Models/load_shifting_ppo_Washing_Machine")

print(evaluate_policy(model,env,n_eval_episodes=100))

# Evaluate the model
obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward

best_action, best_reward = env.get_best_action()
print(f"Best Action (Time step to start washing machine): {best_action/60} hour")
print(f"Best Reward: {best_reward}")




'''

log_path = os.path.join('Training','Logs')
save_path = os.path.join('Training', 'Saved Models')

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=50, verbose=1)
eval_callback = EvalCallback(env, ########## Saving the best possible model which has a score of more than 200 . Fot this environment 200 is seen as a good score. ep_len_mean  is the score
                             callback_on_new_best=stop_callback,
                             eval_freq=100000,
                             best_model_save_path=save_path,
                             verbose=1)


model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000,callback=eval_callback)


'''
