import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta


FILE_DIR_PATH = Path(__file__).parent.parent.parent

Efficiency_list = [10 ,20 ,30 ,40 ,50 ,60 ,70 ,80 ,90 ,100]

df_total_power = pd.DataFrame()

for Efficiency in Efficiency_list:

    #Best_action = pd.read_csv(FILE_DIR_PATH/f'DRL_model/Best_Action/NILM_efficiency/NILM_efficiency_Best_Action_{Efficiency}.csv',sep=';')
    #DRL_Input = pd.read_csv(FILE_DIR_PATH/f'NILM_model/NILM_efficiency/Databases/DRL_input/DRL_NILM_{Efficiency}.csv',sep=';')
    Actual_Load_Profile = pd.read_csv(FILE_DIR_PATH/f'NILM_model/NILM_efficiency/Databases/Load_Profile_actual.csv',sep=';')
    Actual_Load_Profile = Actual_Load_Profile.loc[:, ~Actual_Load_Profile.columns.str.contains('^Unnamed')]
    Actual_Load_Profile = Actual_Load_Profile.drop(columns=['Total_Power'])

    start_time = datetime(2021, 1, 1, 0, 0, 0)
    end_time = datetime(2021, 1, 1, 23, 59, 0)

    delta = timedelta(days=1)
    date_range = [start_time + timedelta(days=i) for i in range((end_time - start_time).days + 1)]
    data_frames = []

    for date in date_range:

        Power_run_time = pd.read_csv(FILE_DIR_PATH/f'NILM_model/NILM_efficiency/Databases/DRL_input/DRL_NILM_{Efficiency}.csv',sep=';')
        Turning_on = pd.read_csv(FILE_DIR_PATH/f'DRL_model/Best_Action/NILM_efficiency/NILM_efficiency_Best_Action_{Efficiency}.csv',sep=';')


        Dish_Washer_Run_Time             = Power_run_time['Dish_Washer'][0]
        Kettle_Run_Time                  = Power_run_time['Kettle'][0]
        Toaster_Run_Time                 = Power_run_time['Toaster'][0]
        Vaccum_Cleaner_Run_Time          = Power_run_time['Vaccum_Cleaner'][0]
        Clothing_Iron_Run_Time           = Power_run_time['Clothing_Iron'][0]
        Oven_Run_Time                    = Power_run_time['Oven'][0]
        EV_Run_Time                      = Power_run_time['EV'][0]
        Electric_Water_Heater_Run_Time   = Power_run_time['Electric_Water_Heater'][0]
        Wash_Dryer_Run_Time              = Power_run_time['Wash_Dryer'][0]
        Washing_machine_Run_Time         = Power_run_time['Washing_Machine'][0]


        Dish_Washer_Power            = Power_run_time['Dish_Washer'][1]
        Kettle_Power                 = Power_run_time['Kettle'][1]
        Toaster_Power                = Power_run_time['Toaster'][1]
        Vaccum_Cleaner_Power             = Power_run_time['Vaccum_Cleaner'][1]
        Clothing_Iron_Power              = Power_run_time['Clothing_Iron'][1]
        Oven_Power                   = Power_run_time['Oven'][1]
        EV_Power                     = Power_run_time['EV'][1]
        Electric_Water_Heater_Power      = Power_run_time['Electric_Water_Heater'][1]
        Wash_Dryer_Power                 = Power_run_time['Wash_Dryer'][1]
        Washing_machine_Power            = Power_run_time['Washing_Machine'][1]


        Dish_Washer_Turning_on           = date + timedelta(minutes=int(Turning_on['Dish_Washer'][0] * 60))
        Kettle_Turning_on                = date + timedelta(minutes=int(Turning_on['Kettle'][0] * 60))
        Toaster_Turning_on               = date + timedelta(minutes=int(Turning_on['Toaster'][0] * 60))
        Vaccum_Cleaner_Turning_on        = date + timedelta(minutes=int(Turning_on['Vaccum_Cleaner'][0] * 60))
        Clothing_Iron_Turning_on         = date + timedelta(minutes=int(Turning_on['Clothing_Iron'][0] * 60))
        Oven_Turning_on                  = date + timedelta(minutes=int(Turning_on['Oven'][0] * 60))
        EV_Turning_on                    = date + timedelta(minutes=int(Turning_on['EV'][0] * 60 ))
        Electric_Water_Heater_Turning_on = date + timedelta(minutes=int(Turning_on['Electric_Water_Heater'][0] * 60 ))
        Wash_Dryer_Turning_on            = date + timedelta(minutes=int(Turning_on['Wash_Dryer'][0] * 60))
        Washing_machine_Turning_on       = date + timedelta(minutes=int(Turning_on['Washing_Machine'][0] * 60 ))


        Start_day = date
        End_day = date + timedelta(minutes=1439)
        time_index = pd.date_range(start=Start_day, end=End_day, freq='T')
        df = pd.DataFrame(index=time_index)

        appliances = [ 'Dish_Washer' ,'Kettle' ,'Toaster' ,'Vaccum_Cleaner' ,'Clothing_Iron' ,'Oven', 'EV' , 'Electric_Water_Heater' , 'Wash_Dryer' , 'Washing_Machine' ]
        for appliance in appliances:
            df[appliance] = 0.0

        def fill_appliance_load(df, appliance_name, start_time, runtime, power):

            end_time = start_time + timedelta(minutes=int(runtime) - 1)
            df.loc[start_time:end_time, appliance_name] = power


        fill_appliance_load(df, 'Dish_Washer', Dish_Washer_Turning_on, Dish_Washer_Run_Time, Dish_Washer_Power)
        fill_appliance_load(df, 'Kettle', Kettle_Turning_on, Kettle_Run_Time, Kettle_Power)
        fill_appliance_load(df, 'Toaster', Toaster_Turning_on, Toaster_Run_Time, Toaster_Power)
        fill_appliance_load(df, 'Vaccum_Cleaner', Vaccum_Cleaner_Turning_on, Vaccum_Cleaner_Run_Time, Vaccum_Cleaner_Power)
        fill_appliance_load(df, 'Clothing_Iron', Clothing_Iron_Turning_on, Clothing_Iron_Run_Time, Clothing_Iron_Power)
        fill_appliance_load(df, 'Oven', Oven_Turning_on, Oven_Run_Time, Oven_Power)
        fill_appliance_load(df, 'EV', EV_Turning_on, EV_Run_Time, EV_Power)
        fill_appliance_load(df, 'Electric_Water_Heater', Electric_Water_Heater_Turning_on, Electric_Water_Heater_Run_Time,Electric_Water_Heater_Power)
        fill_appliance_load(df, 'Wash_Dryer', Wash_Dryer_Turning_on, Wash_Dryer_Run_Time, Wash_Dryer_Power)
        fill_appliance_load(df, 'Washing_Machine', Washing_machine_Turning_on, Washing_machine_Run_Time,Washing_machine_Power)

        data_frames.append(df)

    final_df = pd.concat(data_frames)

    final_df = final_df.reset_index()
    final_df = final_df.drop(columns=['index'])
    final_df = final_df.loc[:, (final_df != 0).any(axis=0)]

    Actual_Load_Profile[final_df.columns] = final_df
    Actual_Load_Profile['Total_Power'] = Actual_Load_Profile.sum(axis=1)

    Actual_Load_Profile.to_csv(FILE_DIR_PATH/f'NILM_model/NILM_efficiency/Databases/Load_Profile_{Efficiency}.csv',sep=';')

    Total_Power = Actual_Load_Profile['Total_Power']
    df_total_power[f'Total_Power_{Efficiency}'] = Total_Power


df_total_power.to_csv(FILE_DIR_PATH/f'NILM_model/NILM_efficiency/Databases/Total_Power_all_efficiencies_1_min.csv',sep=';')


def reducing_resolution(df, chunk_size):
    num_chunks = len(df) // chunk_size

    averages = []
    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size]
        chunk_avg = chunk.mean()
        averages.append(chunk_avg)
    averages_df = pd.DataFrame(averages)

    return averages_df

averages_df = reducing_resolution(df_total_power,15)

averages_df.to_csv(FILE_DIR_PATH/f'NILM_model/NILM_efficiency/Databases/Total_Power_all_efficiencies_15_min.csv',sep=';')