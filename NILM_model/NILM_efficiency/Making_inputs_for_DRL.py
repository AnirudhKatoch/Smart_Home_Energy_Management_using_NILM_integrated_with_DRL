import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FILE_DIR_PATH = Path(__file__).parent.parent

df = pd.DataFrame()

def DRL_Input_make(df ,power ,duration ,column_name ):

    df[column_name] = 0
    df.at[0, column_name] = duration
    df.at[1, column_name] = power

    return df


df = DRL_Input_make(df, power='Power', duration='Run_Time', column_name='Specifications')
df = DRL_Input_make(df, power=0, duration=0, column_name='Dish_Washer')
df = DRL_Input_make(df, power=0, duration=0, column_name='Kettle')
df = DRL_Input_make(df, power=0, duration=0, column_name='Toaster')
df = DRL_Input_make(df, power=0, duration=0, column_name='Vaccum_Cleaner')
df = DRL_Input_make(df, power=0, duration=0, column_name='Clothing_Iron')
df = DRL_Input_make(df, power=0, duration=0, column_name='Oven')
df = DRL_Input_make(df, power=0, duration=0, column_name='EV')
df = DRL_Input_make(df, power=0, duration=0, column_name='Electric_Water_Heater')
df = DRL_Input_make(df, power=0, duration=0, column_name='Wash_Dryer')
df = DRL_Input_make(df, power=0, duration=0, column_name='Washing_Machine')


df.to_csv(FILE_DIR_PATH/f'NILM_efficiency/Databases/DRL_input/DRL_NILM_0.csv',sep=';')
