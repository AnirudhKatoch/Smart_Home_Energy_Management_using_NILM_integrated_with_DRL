import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FILE_DIR_PATH = Path(__file__).parent.parent


df = pd.read_csv(FILE_DIR_PATH/'Database/PV_elec/1_min_resolution/DRL_1_day_mean_inputs.csv',sep=';')

df['Washing_Machine'] = 0
df.loc[int(17 * 60) : int(17 * 60 + 60), 'Washing_Machine'] = 1000/2

df['Cloth_Dryer'] = 0
df.loc[int(19 * 60) : int(19 * 60 + 60), 'Cloth_Dryer'] = 3000/2

df['Dish_Washer'] = 0
df.loc[int(21 * 60) : int(21 * 60 + 120), 'Dish_Washer'] = 2000/2

df['EV'] = 0
df.loc[int(18 * 60) : int(18 * 60 + 120), 'EV'] = 5000/2

df['Electric_Water_Heater'] = 0
df.loc[int(6 * 60) : int(6 * 60 + 90), 'Electric_Water_Heater'] = 4000/2

df['Load'] = df['Washing_Machine'] + df['Cloth_Dryer'] + df['Dish_Washer'] + df['EV'] + df['Electric_Water_Heater']

df['Load'] = df['Load'] /60 + 500/60

print(df['Load'].sum()*365/(1000000))