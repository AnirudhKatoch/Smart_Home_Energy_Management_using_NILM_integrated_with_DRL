import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

FILE_DIR_PATH = Path(__file__).parent.parent

Load_Profile = 'Load_Profile_for_HIL'

start_time = datetime(2021, 1, 1, 0, 0, 0)
end_time = datetime(2021, 1, 1, 23, 59, 0)

delta = timedelta(days=1)

date_range = [start_time + timedelta(days=i) for i in range((end_time - start_time).days + 1)]

data_frames = []

for date in date_range:

    day_index = date.strftime('%Y-%m-%d 00:00:00')
    day = date.strftime('%Y-%m-%d %H-%M-%S')
    month = date.strftime('%Y-%m')

    Power_run_time = pd.read_csv(FILE_DIR_PATH / f'NILM_model/Results/Database_Table_For_DRL/{Load_Profile}/{month}.csv', sep=';')
    Turning_on = pd.read_csv(FILE_DIR_PATH / f'DRL_model/Best_Action/{Load_Profile}/{day}.csv', sep=';')

    Washing_machine_Run_Time = Power_run_time['Washing_Machine'][0]
    Wash_Dryer_Run_Time = Power_run_time['Wash_Dryer'][0]
    Dish_Washer_Run_Time = Power_run_time['Dish_Washer'][0]
    EV_Run_Time = Power_run_time['EV'][0]
    Electric_Water_Heater_Run_Time = Power_run_time['Electric_Water_Heater'][0]
    Washing_Machine_2_Run_Time = Power_run_time['Washing_Machine_2'][0]

    Washing_machine_Power = Power_run_time['Washing_Machine'][1]
    Wash_Dryer_Power = Power_run_time['Wash_Dryer'][1]
    Dish_Washer_Power = Power_run_time['Dish_Washer'][1]
    EV_Power = Power_run_time['EV'][1]
    Electric_Water_Heater_Power = Power_run_time['Electric_Water_Heater'][1]
    Washing_Machine_2_Power = Power_run_time['Washing_Machine_2'][1]

    Washing_machine_Turning_on = date + timedelta(minutes=int( Turning_on['Washing_Machine'][0] * 60 ))
    Wash_Dryer_Turning_on = date + timedelta(minutes=int( Turning_on['Wash_Dryer'][0] * 60 ))
    Dish_Washer_Turning_on = date + timedelta(minutes=int( Turning_on['Dish_Washer'][0] * 60 ))
    EV_Turning_on = date + timedelta(minutes=int( Turning_on['EV'][0] * 60 ))
    Electric_Water_Heater_Turning_on = date + timedelta(minutes=int( Turning_on['Electric_Water_Heater'][0] * 60 ))
    Washing_Machine_2_Turning_on = date + timedelta(minutes=int( Turning_on['Washing_Machine_2'][0] * 60 ))

    Start_day = date
    End_day = date + timedelta(minutes=1439)
    time_index = pd.date_range(start=Start_day, end=End_day, freq='T')
    df = pd.DataFrame(index=time_index)

    appliances = ['Washing_Machine', 'Wash_Dryer', 'Dish_Washer', 'EV', 'Electric_Water_Heater', 'Washing_Machine_2']
    for appliance in appliances:
        df[appliance] = 0.0

    def fill_appliance_load(df, appliance_name, start_time, runtime, power):

        if runtime == 0:
            power = 0

        end_time = start_time + timedelta(minutes=runtime-1)
        df.loc[start_time:end_time, appliance_name] = power

    fill_appliance_load(df, 'Washing_Machine', Washing_machine_Turning_on, Washing_machine_Run_Time,Washing_machine_Power)
    fill_appliance_load(df, 'Wash_Dryer', Wash_Dryer_Turning_on, Wash_Dryer_Run_Time, Wash_Dryer_Power)
    fill_appliance_load(df, 'Dish_Washer', Dish_Washer_Turning_on, Dish_Washer_Run_Time, Dish_Washer_Power)
    fill_appliance_load(df, 'EV', EV_Turning_on, EV_Run_Time, EV_Power)
    fill_appliance_load(df, 'Electric_Water_Heater', Electric_Water_Heater_Turning_on, Electric_Water_Heater_Run_Time,Electric_Water_Heater_Power)
    fill_appliance_load(df, 'Washing_Machine_2', Washing_Machine_2_Turning_on, Washing_Machine_2_Run_Time,Washing_Machine_2_Power)

    data_frames.append(df)

final_df = pd.concat(data_frames)

df_predicted_ohne = pd.read_csv(FILE_DIR_PATH/f'NILM_model/Results/{Load_Profile}/Predicted_without_load_shifters.csv',sep=';')
df_predicted_ohne = df_predicted_ohne.loc[:, ~df_predicted_ohne.columns.str.contains('^Unnamed')]
df_predicted_ohne = df_predicted_ohne[:len(final_df)]

final_df = final_df.reset_index()
final_df = final_df.loc[:, ~final_df.columns.str.contains('^index')]

df_optimized_predicted = df_predicted_ohne.join(final_df)

df_optimized_predicted['Total_Power'] = df_optimized_predicted.sum(axis=1)



df_optimized_predicted.to_csv(FILE_DIR_PATH / f'NILM_model/Results/{Load_Profile}/Optimized_predicted_load_profile.csv', sep=';',index=False)

print(df_optimized_predicted)

