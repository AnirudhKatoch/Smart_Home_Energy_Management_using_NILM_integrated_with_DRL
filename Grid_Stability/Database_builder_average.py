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

def average_formation(df,index_before, index_after):

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
    sum_actual = stacked_actual.mean(axis=1)
    sum_predicted = stacked_predicted.mean(axis=1)

    return sum_actual*20, sum_predicted*20

Load_average_actual, Load_average_optimized = average_formation(df,index_before="2021-01-01 00:00:00", index_after="2021-12-31 23:45:00")

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

PV_average = PV_formation(PV_Power,index_before="2021-01-01 00:00:00", index_after="2021-12-31 23:59:59")

df_new = pd.DataFrame()

df_new['PV_average'] = PV_average
df_new['Load_average_actual'] = Load_average_actual
df_new['Load_average_optimized'] = Load_average_optimized

df_new.to_csv(FILE_DIR_PATH/f'Grid_Stability/Database/Average_inputs.csv',sep=';')




plt.figure(figsize=(6.4, 4.8))
plt.plot(df_new.index/4, df_new['PV_average']/100, label='Irradiance')
plt.xlabel('Time (h)', fontsize=15)
plt.ylabel('Irradiance (W/m2))', fontsize=15)
plt.xlim(min(df_new.index/4),max(df_new.index/4))
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.legend(fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig(FILE_DIR_PATH/f'Grid_Stability/Figures/Irradiance_for_building.png')

'''

# Plot Load data (actual) for each season
plt.figure(figsize=(6.4, 4.8))
plt.plot(df_new.index/4, df_new['Load_average_actual']/1000, label='Scenario 1')
plt.plot(df_new.index/4, df_new['Load_average_optimized']/1000, label='Scenario 2')
plt.xlabel('Time (h)', fontsize=15)
plt.ylabel('Power (kW))', fontsize=15)
plt.xlim(min(df_new.index/4),max(df_new.index/4))
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.legend(fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig(FILE_DIR_PATH/f'Grid_Stability/Figures/Building_load.png')

'''