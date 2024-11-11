import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from Environment_class import LoadShiftingEnv, SaveRewardsCallback

FILE_DIR_PATH = Path(__file__).parent.parent

# Replace colons with dashes for valid filename
def safe_filename(timestamp):
    return timestamp.replace(':', '-')

Load_Profile = 'Load_Profile_2'

df = pd.read_csv(FILE_DIR_PATH/f'PV_Grid_Suppy_and_Feed_in_inputs/DRL_input_database.csv',sep=';')
df.index = df['Unnamed: 0']
df = df.drop(columns=['Unnamed: 0'])
df = df.rename_axis(index=None)

date_range = pd.date_range(start='2021-01-01', end='2021-12-31', freq='D')
#date_range = pd.date_range(start='2021-07-01', end='2021-07-02', freq='D')

# Iterate over each day in the range
for single_date in date_range:

    start_time = single_date.strftime('%Y-%m-%d 00:00:00')
    end_time = single_date.strftime('%Y-%m-%d 23:59:00')

    # Extracting data for each day
    PV_power = np.array(df['PV_Power_W'][start_time:end_time])
    Electricity_price = np.array(df['Supply_price_EUR_Wh'][start_time:end_time])
    Feed_in_tariff = np.array(df['Feed_in_Tariff_EUR_Wh'][start_time:end_time])

    month = single_date.strftime('%Y-%m')
    Appliance_specifications = pd.read_csv(FILE_DIR_PATH/f'NILM_model/Results/Database_Table_For_DRL/{Load_Profile}/{month}.csv',sep=';')

    safe_start_time = safe_filename(start_time)

    env = LoadShiftingEnv(PV_power, Electricity_price, Feed_in_tariff, Appliance_specifications)
    model = PPO("MlpPolicy", env, verbose=1)
    save_episode_reward_callback = SaveRewardsCallback(file_path=FILE_DIR_PATH/f'DRL_model/Rewards/{Load_Profile}/{safe_start_time}.csv')
    model.learn(total_timesteps=50000, callback=save_episode_reward_callback)
    model.save(f"Models/{Load_Profile}/{safe_start_time}")
    del model
    model = PPO.load(f"Models/{Load_Profile}/{safe_start_time}")

    evaluate_policy_results = evaluate_policy(model, env, n_eval_episodes=100)

    # Evaluate the model
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    best_action, best_reward = env.get_best_action()

    df_action = pd.DataFrame({
        'Washing_Machine': [best_action[0] / 60],
        'Wash_Dryer': [best_action[1] / 60],
        'Dish_Washer': [best_action[2] / 60],
        'EV': [best_action[3] / 60],
        'Electric_Water_Heater': [best_action[4] / 60],
        'Washing_Machine_2': [best_action[5] / 60],
        'Best_Reward': [best_reward],
        'evaluate_policy_results': [evaluate_policy_results]
    })

    df_action.to_csv(FILE_DIR_PATH/f'DRL_model/Best_Action/{Load_Profile}/{safe_start_time}.csv', sep=';')