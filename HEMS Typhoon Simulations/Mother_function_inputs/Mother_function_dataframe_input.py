import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FILE_DIR_PATH = Path(__file__).parent.parent.parent.parent.parent

PV_Irradiance = pd.read_csv(FILE_DIR_PATH/f'Solar PV/Faster PV/Freiburg_2021_PV.csv',sep=';')
PV_Irradiance = (PV_Irradiance['GS_10']).tolist()

Load_actual_kW = pd.read_csv(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/NILM_model/Results/Load_Profile_2/Actual.csv',sep=';')
Load_actual_kW = Load_actual_kW.groupby(Load_actual_kW.index // 15).mean()
Load_actual_kW = (Load_actual_kW['Total_Power']/1000).tolist()

Load_Optimized_predicted_kW = pd.read_csv(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/NILM_model/Results/Load_Profile_2/Optimized_predicted_load_profile.csv',sep=';')
Load_Optimized_predicted_kW = Load_Optimized_predicted_kW.groupby(Load_Optimized_predicted_kW.index // 15).mean()
Load_Optimized_predicted_kW = (Load_Optimized_predicted_kW['Total_Power']/1000).tolist()

Time_unix = pd.read_csv(FILE_DIR_PATH / f'NILM_and_DRL_based_EMS/DESO/Time stamp/timestamp_DESO_15_min_2021.csv',sep=';')
Time_unix = Time_unix['timestamps'].tolist()


df = pd.DataFrame()


df['PV_Irradiance'] = PV_Irradiance
df['Load_actual_kW'] = Load_actual_kW
df['Load_Optimized_predicted_kW'] = Load_Optimized_predicted_kW
df['Time_unix'] = Time_unix

df.to_csv(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',sep=';')