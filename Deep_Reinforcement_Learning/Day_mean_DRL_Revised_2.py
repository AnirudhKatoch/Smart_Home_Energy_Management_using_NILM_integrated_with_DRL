import gym
from gym import Env
from gym import spaces
import numpy as np
from pathlib import Path
import pandas as pd


FILE_DIR_PATH = Path(__file__).parent.parent

class LoadShiftingEnv(Env):


    def __init__(self, pv_power, electricity_price, feed_in_tariff):

        super(LoadShiftingEnv, self).__init__()

        self.pv_power = pv_power
        self.electricity_price = electricity_price
        self.feed_in_tariff = feed_in_tariff
        self.load_power = 1000
        self.load_duration = 6
        self.time_steps = len(pv_power)

        # Action space: binary, 0 (don't run) or 1 (run the washing machine)
        self.action_space = spaces.Discrete(2)

        # Observation space: current time step, PV power, electricity price, feed-in tariff, remaining load steps
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)


    def step(self, action):

        reward = 0

        Power = self.pv_power[self.current_step] - self.pv_power[self.load_power]
        if Power >= 0:
            Energy = (Power/4) * self.feed_in_tariff[self.current_step]
        if Power < 0:
            Energy = (Power / 4) * self.feed_in_tariff[self.current_step]



        self.total_reward += reward
        self.current_step += 1

        done = False

        return self._get_observation(), reward, done, {}






    def _get_observation(self):

        return np.array([
            self.current_step,  # Current time step in the simulation
            self.pv_power[self.current_step],  # PV power available at the current time step
            self.electricity_price[self.current_step],  # Cost of electricity from the grid at the current time step
            self.feed_in_tariff[self.current_step], # Price for feeding excess power back to the grid at the current time step
            self.remaining_load_steps  # Number of steps remaining for the washing machine to complete its cycle
        ], dtype=np.float32)


    def render(self):
        pass

    def reset(self):

        self.current_step = 0
        self.remaining_load_steps = self.load_duration
        self.total_reward = 0

        return self._get_observation

df = pd.read_csv(FILE_DIR_PATH / 'Database/PV_elec/mean_96_steps/DRL_1_day_mean_inputs.csv', sep=';')
pv_power = np.array(df['PV_Power_W']) # PV Power (in W)
electricity_price = np.array(df['Electricity_Price_EUR_Wh'])  # Grid electricity price (in $/Wh)
feed_in_tariff = np.array(df['Feed_in_Tariff_EUR_Wh'])  # Feed-in tariff (in $/Wh)

env = LoadShiftingEnv(pv_power, electricity_price, feed_in_tariff)

env.step(0)