from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

FILE_DIR_PATH = Path(__file__).parent.parent

df = pd.read_csv(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',sep=';')
df = df.drop(columns=['Unnamed: 0', 'PV_Irradiance','Time_unix'])

df['Load_actual_kW'] = df['Load_actual_kW'] * 1000
df['Load_Optimized_predicted_kW'] = df['Load_Optimized_predicted_kW'] * 1000

date_range = pd.date_range(start="2021-01-01 00:00:00", end="2021-12-31 23:45:00", freq="15T")
df.index = date_range

def seasons_formation(df,index_before, index_after):

    dataframe_season = df.loc[index_before:index_after]

    window_size = 96
    num_windows = len(dataframe_season) // 96

    windows_actual = []
    windows_predicted = []

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        windows_actual.append(dataframe_season['Load_actual_kW'].iloc[start_idx:end_idx].reset_index(drop=True))
        windows_predicted.append(dataframe_season['Load_Optimized_predicted_kW'].iloc[start_idx:end_idx].reset_index(drop=True))

    stacked_actual = pd.concat(windows_actual, axis=1)
    stacked_predicted = pd.concat(windows_predicted, axis=1)
    sum_actual = stacked_actual.sum(axis=1)
    sum_predicted = stacked_predicted.sum(axis=1)

    return sum_actual*2, sum_predicted*2

Load_Winter_actual, Load_Winter_optimized = seasons_formation(df,index_before="2021-01-11", index_after="2021-01-20")
Load_Spring_actual, Load_Spring_optimized = seasons_formation(df,index_before="2021-04-11", index_after="2021-04-20")
Load_Summer_actual, Load_Summer_optimized = seasons_formation(df,index_before="2021-07-11", index_after="2021-07-20")
Load_Autumn_actual, Load_Autumn_optimized = seasons_formation(df,index_before="2021-10-11", index_after="2021-10-20")

df_PV = pd.read_feather(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Actual/Real/df_absolute.feather')
PV_Power = df_PV['PV Power Plant (Generic) UI1.Pmeas_kW']
date_range_second = pd.date_range(start="2021-01-01 00:00:00", end="2021-12-31 23:59:59", freq="1s")
PV_Power.index = date_range_second

def PV_formation(PV_Power,index_before, index_after):

    dataframe_PV= PV_Power.loc[index_before:index_after]

    window_size = 86400
    num_windows = len(dataframe_PV) // 86400

    windows_actual = []

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        windows_actual.append(dataframe_PV.iloc[start_idx:end_idx].reset_index(drop=True))

    stacked_actual = pd.concat(windows_actual, axis=1)
    sum_actual = stacked_actual.mean(axis=1)

    sum_actual = sum_actual.groupby(np.arange(len(sum_actual)) // 900).mean()

    return sum_actual*10

PV_Winter = PV_formation(PV_Power,index_before="2021-01-11 00:00:00", index_after="2021-01-20 23:59:59")
PV_Spring = PV_formation(PV_Power,index_before="2021-04-11 00:00:00", index_after="2021-04-20 23:59:59")
PV_Summer = PV_formation(PV_Power,index_before="2021-07-11 00:00:00", index_after="2021-07-20 23:59:59")
PV_Autumn = PV_formation(PV_Power,index_before="2021-10-11 00:00:00", index_after="2021-10-20 23:59:59")

df_new = pd.DataFrame()

df_new['PV_Winter'] = PV_Winter
df_new['PV_Spring'] = PV_Spring
df_new['PV_Summer'] = PV_Summer
df_new['PV_Autumn'] = PV_Autumn

df_new['Load_Winter_actual'] = Load_Winter_actual
df_new['Load_Spring_actual'] = Load_Spring_actual
df_new['Load_Summer_actual'] = Load_Summer_actual
df_new['Load_Autumn_actual'] = Load_Autumn_actual

df_new['Load_Winter_optimized'] = Load_Winter_optimized
df_new['Load_Spring_optimized'] = Load_Spring_optimized
df_new['Load_Summer_optimized'] = Load_Summer_optimized
df_new['Load_Autumn_optimized'] = Load_Autumn_optimized

df_new.to_csv(FILE_DIR_PATH/f'Grid_Stability/Database/Seasonal_variation_inputs.csv',sep=';')

# Plot PV data for each season
plt.figure(figsize=(12, 6))
plt.plot(df_new.index, df_new['PV_Winter'], label='PV Winter', alpha=0.7)
plt.plot(df_new.index, df_new['PV_Spring'], label='PV Spring', alpha=0.7)
plt.plot(df_new.index, df_new['PV_Summer'], label='PV Summer', alpha=0.7)
plt.plot(df_new.index, df_new['PV_Autumn'], label='PV Autumn', alpha=0.7)
plt.title("PV Power Across Seasons")
plt.xlabel("Index")
plt.ylabel("PV Power")
plt.legend()
plt.grid(True)

# Plot Load data (actual) for each season
plt.figure(figsize=(12, 6))
plt.plot(df_new.index, df_new['Load_Winter_actual'], label='Load Winter Actual', alpha=0.7)
plt.plot(df_new.index, df_new['Load_Spring_actual'], label='Load Spring Actual', alpha=0.7)
plt.plot(df_new.index, df_new['Load_Summer_actual'], label='Load Summer Actual', alpha=0.7)
plt.plot(df_new.index, df_new['Load_Autumn_actual'], label='Load Autumn Actual', alpha=0.7)
plt.title("Actual Load Across Seasons")
plt.xlabel("Index")
plt.ylabel("Load (Actual)")
plt.legend()
plt.grid(True)

# Plot Load data (optimized) for each season
plt.figure(figsize=(12, 6))
plt.plot(df_new.index, df_new['Load_Winter_optimized'], label='Load Winter Optimized', alpha=0.7)
plt.plot(df_new.index, df_new['Load_Spring_optimized'], label='Load Spring Optimized', alpha=0.7)
plt.plot(df_new.index, df_new['Load_Summer_optimized'], label='Load Summer Optimized', alpha=0.7)
plt.plot(df_new.index, df_new['Load_Autumn_optimized'], label='Load Autumn Optimized', alpha=0.7)
plt.title("Optimized Load Across Seasons")
plt.xlabel("Index")
plt.ylabel("Load (Optimized)")
plt.legend()
plt.grid(True)

plt.show()
