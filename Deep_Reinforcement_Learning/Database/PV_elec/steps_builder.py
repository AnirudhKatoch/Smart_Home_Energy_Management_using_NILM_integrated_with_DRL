import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

FILE_DIR_PATH = Path(__file__).parent.parent.parent.parent


Electricity_Price = pd.read_csv(FILE_DIR_PATH/'Deep_Reinforcement_Learning/Database/PV_elec/mean_86400_steps/Electricity_Price.csv',sep=';')
Feed_in_Tariff = pd.read_csv(FILE_DIR_PATH/'Deep_Reinforcement_Learning/Database/PV_elec/mean_86400_steps/Feed_in_Tariff_EUR_kWh.csv',sep=';')
PV_Power = pd.read_csv(FILE_DIR_PATH/'z/Standard_deviation_weg/csv_results/synPRO_el_family/Digital_Twin/mean/df_absolute.csv',sep=';')




df = pd.DataFrame()

df['Electricity_Price_EUR_Wh'] = Electricity_Price['Electricity_Price_mean']/1000
df['Feed_in_Tariff_EUR_Wh'] = Feed_in_Tariff['Tariff_mean']/1000
df['PV_Power_W']  = PV_Power['PV Power Plant (Generic) UI1.Pmeas_kW']

chunk_size = 60
num_chunks = len(df) // chunk_size

averages = []
for i in range(num_chunks):
    chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size]
    chunk_avg = chunk.mean()
    averages.append(chunk_avg)
averages_df = pd.DataFrame(averages)


print(max(df['Electricity_Price_EUR_Wh'])*1000)



#averages_df.to_csv(FILE_DIR_PATH/'Deep_Reinforcement_Learning/Database/PV_elec/1_min_resolution/DRL_1_day_mean_inputs.csv',sep=';')