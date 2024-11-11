import gym
from gym import spaces
import numpy as np
from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import os

FILE_DIR_PATH = Path(__file__).parent.parent

class LoadShiftingEnv(gym.Env):

    def __init__(self, pv_power, electricity_price, feed_in_tariff, appliance_table):

        super(LoadShiftingEnv, self).__init__()

        # Inputs: PV power, electricity price, and feed-in price (each has 1440 steps (1 minute resolution))

        self.pv_power = pv_power
        self.electricity_price = electricity_price
        self.feed_in_tariff = feed_in_tariff
        self.total_steps = len(pv_power)

        self.Washing_machine_run_time = appliance_table['Washing_Machine'][0]
        self.Washing_machine_power = appliance_table['Washing_Machine'][1]

        self.Cloth_Dryer_run_time = appliance_table['Cloth_Dryer'][0]
        self.Cloth_Dryer_power = appliance_table['Cloth_Dryer'][1]

        self.Dish_Washer_run_time = appliance_table['Dish_Washer'][0]
        self.Dish_Washer_power = appliance_table['Dish_Washer'][1]

        self.EV_run_time = appliance_table['EV'][0]
        self.EV_power = appliance_table['EV'][1]

        self.Electric_Water_Heater_run_time = appliance_table['Electric_Water_Heater'][0]
        self.Electric_Water_Heater_power = appliance_table['Electric_Water_Heater'][1]

        # Action space: choose start times for both washing machine and EV
        self.action_space = spaces.MultiDiscrete([self.total_steps - self.Washing_machine_run_time,
                                                  self.total_steps - self.Cloth_Dryer_run_time,
                                                  self.total_steps - self.Dish_Washer_run_time,
                                                  self.total_steps - self.EV_run_time,
                                                  self.total_steps - self.Electric_Water_Heater_run_time])

        # Observation space: [time step, pv power, electricity price, feed-in price]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)

        # Initialize the environment
        self.reset()

    def reset(self):

        self.current_step = 0

        self.Washing_Machine_start = 0
        self.Cloth_Dryer_start = 0
        self.Dish_Washer_start = 0
        self.EV_start = 0
        self.Electric_Water_Heater_start = 0

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

        self.Washing_Machine_start        = action[0] + 1
        self.Cloth_Dryer_start            = action[1] + 1
        self.Dish_Washer_start            = action[2] + 1
        self.EV_start                     = action[3] + 1
        self.Electric_Water_Heater_start  = action[4] + 1

        total_feed_in = 0
        total_supply_cost = 0

        for i in range(self.total_steps):

            net_power = self.pv_power[i]

            # Washing machine consumption
            if self.Washing_Machine_start <= i + 1 < self.Washing_Machine_start + self.Washing_machine_run_time:
                net_power -= self.Washing_machine_power

            # Cloth Dryer consumption
            if self.Cloth_Dryer_start <= i + 1 < self.Cloth_Dryer_start + self.Cloth_Dryer_run_time:
                net_power -= self.Cloth_Dryer_power

            # Dish Washer consumption
            if self.Dish_Washer_start <= i + 1 < self.Dish_Washer_start + self.Dish_Washer_run_time:
                net_power -= self.Dish_Washer_power

            # EV consumption
            if self.EV_start <= i + 1 < self.EV_start + self.EV_run_time:
                net_power -= self.EV_power

            # Electric Water Heater consumption
            if self.Electric_Water_Heater_start <= i + 1 < self.Electric_Water_Heater_start + self.Electric_Water_Heater_run_time:
                net_power -= self.Electric_Water_Heater_power

            if net_power > 0:
                total_feed_in += net_power * (1/60) * self.feed_in_tariff[i]

            elif net_power < 0:
                total_supply_cost += abs(net_power) * (1/60) * self.electricity_price[i]

            elif net_power == 0:
                total_feed_in += 0
                total_supply_cost += 0

        total_reward = total_feed_in - total_supply_cost

        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_action = action

        self.done = True

        info = {'total_reward': total_reward}

        return self._next_observation(), total_reward, self.done, info

    def get_best_action(self):
        return self.best_action, self.best_reward

    def render(self, mode='human'):
        pass

    def close(self):
        pass

class SaveRewardsCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(SaveRewardsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.mean_rewards = []

    def _on_step(self) -> bool:
        # Check if the episode has ended
        if self.locals['dones'][0]:
            episode_reward = self.locals['infos'][0]['total_reward']  # Get cumulative reward from info
            self.episode_rewards.append(episode_reward)
        return True

    def _on_rollout_end(self) -> None:
        mean_reward = np.mean(self.locals["rollout_buffer"].rewards)
        self.mean_rewards.append(mean_reward)

    def on_training_end(self):
        # Save the episode rewards to a CSV file
        df_episode = pd.DataFrame(self.episode_rewards, columns=["episode_reward"])
        df_episode.to_csv(FILE_DIR_PATH / "Trial_1/Results/episode_rewards_all_appliances.csv", index=False)

        # Save the mean rewards to a CSV file
        df_mean = pd.DataFrame(self.mean_rewards, columns=["ep_rew_mean"])
        #df_mean.to_csv(FILE_DIR_PATH / "Trial_1/Results/mean_episode_rewards_all_appliances.csv", index=False)



# Load your data
df = pd.read_csv(FILE_DIR_PATH / 'Database/PV_elec/1_min_resolution/DRL_1_day_mean_inputs.csv', sep=';')

pv_power = np.array(df['PV_Power_W']) # PV Power (in W)
electricity_price = np.array(df['Electricity_Price_EUR_Wh'])  # Grid electricity price (in $/Wh)
feed_in_tariff = np.array(df['Feed_in_Tariff_EUR_Wh'])  # Feed-in tariff (in $/Wh)

appliance_table = pd.read_csv(FILE_DIR_PATH/'Database/Appliance_Specifications.csv',sep=';') # Appliance Table

# Create the environment
env = LoadShiftingEnv(pv_power, electricity_price, feed_in_tariff, appliance_table)

# Instantiate the agent
model = A2C("MlpPolicy", env, verbose=1)

# Create the custom callback to save mean episode rewards
save_episode_reward_callback  = SaveRewardsCallback()

# Train the agent
model.learn(total_timesteps = 150000, callback=save_episode_reward_callback)

# Save the model
model.save("Models/load_shifting_ppo_multiple_appliance")

del model

model = A2C.load("Models/load_shifting_ppo_multiple_appliance")

evaluate_policy_results = evaluate_policy(model, env, n_eval_episodes=100)
print('evaluate_policy_results',evaluate_policy_results)

# Evaluate the model
obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward

best_action, best_reward = env.get_best_action()
print(f"Best Action (Time step to start Washing Machine): {best_action[0]/60} hour")
print(f"Best Action (Time step to start Cloth Dryer): {best_action[1]/60} hour")
print(f"Best Action (Time step to start Dish Washer): {best_action[2]/60} hour")
print(f"Best Action (Time step to start EV): {best_action[3]/60} hour")
print(f"Best Action (Time step to start Electric Water Heater): {best_action[4]/60} hour")
print(f"Best Reward: {best_reward}")

df_action = pd.DataFrame({
    'Washing_Machine': [best_action[0] / 60],
    'Cloth_Dryer': [best_action[1] / 60],
    'Dish_Washer': [best_action[2] / 60],
    'EV': [best_action[3] / 60],
    'Electric_Water_Heater': [best_action[4] / 60],
    'Best_Reward': [best_reward],
    'evaluate_policy_results': [evaluate_policy_results]
})

df_action.to_csv(FILE_DIR_PATH/f'Trial_1/Results/Best_action.csv',sep = ';')