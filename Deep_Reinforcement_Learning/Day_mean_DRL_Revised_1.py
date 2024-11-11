import gym
from gym import spaces
import numpy as np
from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

FILE_DIR_PATH = Path(__file__).parent.parent

class LoadShiftingEnv(gym.Env):
    def __init__(self, pv_power, electricity_price, feed_in_tariff):

        super(LoadShiftingEnv, self).__init__()

        self.pv_power = pv_power
        self.electricity_price = electricity_price
        self.feed_in_tariff = feed_in_tariff
        self.load_power = 1000  # Power consumption of the washing machine (W)
        self.load_duration = 6  # Washing machine duration in steps (1.5 hours = 6 x 15 min)
        self.time_steps = len(pv_power) # Total time steps (96 steps for a day)

        # Action space: binary, 0 (don't run) or 1 (run the washing machine)
        self.action_space = spaces.Discrete(2)

        # Observation space: current time step, PV power, electricity price, feed-in tariff, remaining load steps
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

        self.reset()

    def reset(self):

        self.current_step = 0
        self.remaining_load_steps = self.load_duration
        self.total_reward = 0
        return self._get_observation()

    def _get_observation(self):
        return np.array([
            self.current_step,
            self.pv_power[self.current_step-1],
            self.electricity_price[self.current_step-1],
            self.feed_in_tariff[self.current_step-1],
            self.remaining_load_steps
        ], dtype=np.float32)



    def step(self, action):

        reward = 0

        # Execute action only if the current step is valid

        if self.current_step < self.time_steps:

            if action == 1 and self.remaining_load_steps > 0:
                if self.pv_power[self.current_step-1] >= self.load_power:
                    # If PV power is sufficient, fully cover the load and calculate feed-in for excess power
                    excess_pv_power = self.pv_power[self.current_step-1] - self.load_power
                    reward = excess_pv_power * self.feed_in_tariff[self.current_step-1] * 0.25 if excess_pv_power > 0 else 0
                else:
                    # Grid power is required
                    grid_power = self.load_power - self.pv_power[self.current_step-1]
                    reward = -grid_power * self.electricity_price[self.current_step-1] * 0.25  # Cost of grid power

                self.remaining_load_steps -= 1
            else:
                # Feed-in excess PV power to the grid if the washing machine is not running
                excess_pv_power = self.pv_power[self.current_step-1] - self.load_power if self.remaining_load_steps > 0 else self.pv_power[self.current_step-1]
                if excess_pv_power > 0:
                    reward = excess_pv_power * self.feed_in_tariff[self.current_step-1] * 0.25  # Profit from feed-in

            print('start')
            print('reward',reward)
            print('self.current_step',self.current_step)
            #print('self.total_reward',self.total_reward)
            print('end')

            self.total_reward += reward
            self.current_step += 1

        # Ensure the episode runs for 96 steps


        done = self.current_step >= self.time_steps

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):

        print(f'Step: {self.current_step}, Remaining Load Steps: {self.remaining_load_steps}, Total Reward: {self.total_reward} , PV Power : {self.pv_power[self.current_step-1]}')

    def close(self):
        pass



df = pd.read_csv(FILE_DIR_PATH / 'Database/PV_elec/mean_96_steps/DRL_1_day_mean_inputs.csv', sep=';')

pv_power = np.array(df['PV_Power_W']) # PV Power (in W)
electricity_price = np.array(df['Electricity_Price_EUR_Wh'])  # Grid electricity price (in $/Wh)
feed_in_tariff = np.array(df['Feed_in_Tariff_EUR_Wh'])  # Feed-in tariff (in $/Wh)


# Example data
#pv_power = np.random.uniform(0, 2000, 96)  # PV power (W) over 96 time steps
#electricity_price = np.random.uniform(0.1, 0.5, 96)  # Grid electricity price (in $/Wh)
#feed_in_tariff = np.random.uniform(0.05, 0.2, 96)  # Feed-in tariff (in $/Wh)

# Create environment
env = LoadShiftingEnv(pv_power, electricity_price, feed_in_tariff)

# Check if the environment follows Gym's API

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=2500)





# Save the model
model.save("ppo_load_shifting")


# Load the trained model
model = PPO.load("ppo_load_shifting")


# Evaluate the agent
obs = env.reset()
for _ in range(96):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break

env.close()


