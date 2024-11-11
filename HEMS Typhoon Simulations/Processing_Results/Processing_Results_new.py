from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

FILE_DIR_PATH = Path(__file__).parent.parent

def plotting_virtual_HIL(df,csv_path,Title,Save,Show):

    fig1, ax1 = plt.subplots()

    df['Time_seconds'] = df.index/3600

    ax1.plot(df['Time_seconds'], df['PV Power Plant (Generic) UI1.Pmeas_kW']  , label='PV AC Power', color='blue')
    ax1.plot(df['Time_seconds'], df['Three-phase Meter GRID.POWER_P'], label='Grid AC Power', linewidth=1)
    #ax1.plot(df['Time_seconds'], df['Battery DC Power'], label='Battery DC Power')
    ax1.plot(df['Time_seconds'], df['Battery AC Power'] , label='Battery AC Power', color='red',linewidth=3)
    ax1.plot(df['Time_seconds'], df['Variable Load (Generic) UI1.Pmeas_kW'] , label='Load AC Power', color='black',linewidth=1.5)


    ax1.set_xlim(min(df['Time_seconds']), max(df['Time_seconds']))
    ax1.set_xlabel('Hours (h)')
    ax1.set_ylabel('Power (W)')
    ax1.set_title(Title)
    ax1.legend(loc="upper right")
    ax1.grid(True)
    #plt.xticks(range(25))

    # SOC
    fig2, ax2 = plt.subplots()

    ax2.plot(df['Time_seconds'], df['Battery SOC'], label='Battery SOC')
    ax2.set_xlim(min(df['Time_seconds']), max(df['Time_seconds']))
    ax2.set_xlabel('Hours (h)')
    ax2.set_ylabel('SOC (%)')
    ax2.set_title(Title)
    ax2.legend(loc="upper right")
    ax2.grid(True)
    #plt.xticks(range(25))

    if Save == True :
        fig1.savefig(FILE_DIR_PATH / f'{csv_path}/Power_Variation.png')
        fig2.savefig(FILE_DIR_PATH / f'{csv_path}/SOC.png')

    if Show == True:
        plt.show()

def Faster_result_processing(csv_path,Actual_Loop_time, Save):


    def Resolution_change(df, neu_index):

        new_index_size = neu_index
        old_index = np.arange(len(df))

        def interpolate_column(column):
            interpolation_function = CubicSpline(old_index, column)
            new_index = np.linspace(0, len(column) - 1, new_index_size)
            interpolated_values = interpolation_function(new_index)
            return (pd.DataFrame(interpolated_values, columns=[column.name]))

        df = pd.concat([interpolate_column(df[column]) for column in df], axis=1)

        return df

    df = pd.read_feather(FILE_DIR_PATH/f'{csv_path}/df_DESO.feather')

    df['PV Power Plant (Generic) UI1.Available_Ppv_kW'] = df['PV Power Plant (Generic) UI1.Available_Ppv_kW'] * 1000
    df['PV Power Plant (Generic) UI1.Pmeas_kW'] = df['PV Power Plant (Generic) UI1.Pmeas_kW'] * 1000
    df['Variable Load (Generic) UI1.Pmeas_kW'] = df['Variable Load (Generic) UI1.Pmeas_kW'] * 1000
    df['Variable Load (Generic) UI1.Vgrid_rms_meas_kV'] = df['Variable Load (Generic) UI1.Vgrid_rms_meas_kV'] * 1000

    df['time'] = pd.to_timedelta(df['time'])
    df['Time_seconds'] = df['time'].dt.total_seconds()

    df['Time_seconds'] = df['Time_seconds'] - min(df['Time_seconds'])
    df = df.drop(columns=['time'])
    df['Time_seconds'] = df['Time_seconds'].astype(int)

    df = df[df['Time_seconds'] < (Actual_Loop_time)] ############################################################################### Check this value evertime to be sure that the results are correct.

    last_value_count = (df['Time_seconds'] == df['Time_seconds'].iloc[-1]).sum()
    if last_value_count < 10:
        df = df[df['Time_seconds'] != df['Time_seconds'].iloc[-1]]

    unique_time_seconds = df['Time_seconds'].unique()
    processed_dfs = []

    for time_sec in unique_time_seconds:
        print(time_sec)
        df_subset = df[df['Time_seconds'] == time_sec]
        df_subset = Resolution_change(df_subset, 900)
        processed_dfs.append(df_subset)

    df_processed = pd.concat(processed_dfs, ignore_index=True)
    df_processed = df_processed.drop(columns=['Time_seconds'])
    df_processed['Time_seconds'] = df_processed.index
    df = df_processed

    if Save == True:

        df.to_feather(FILE_DIR_PATH/f'{csv_path}/df_absolute_ohne_interpolation.feather')

    return df

def fixing_faster_loop(df, csv_path, Save):

    #df = df[df['Time_seconds'] < (35070.66*900)]

    def Resolution_change(df, neu_index):
        new_index_size = neu_index
        old_index = np.arange(len(df))

        def interpolate_column(column):
            print(column)
            interpolation_function = CubicSpline(old_index, column)
            new_index = np.linspace(0, len(column) - 1, new_index_size)
            interpolated_values = interpolation_function(new_index)
            return (pd.DataFrame(interpolated_values, columns=[column.name]))

        df = pd.concat([interpolate_column(df[column]) for column in df], axis=1)

        return df

    df = Resolution_change(df,3600*365*24)
    df['Time_seconds'] = pd.Index(range(len(df)))

    if Save == True:
        df.to_feather(FILE_DIR_PATH / f'{csv_path}/df_absolute.feather')


    return df

#csv_path = f'SIL_Results/Actual/mit_DESO/Ideal/'
csv_path = f'SIL_Results/Optimized_Predicted/mit_DESO/Ideal/'
#csv_path = f'SIL_Results/Actual/Ideal/'

df = Faster_result_processing(FILE_DIR_PATH/csv_path,Actual_Loop_time = 34943, Save = True)




#df = pd.read_feather(FILE_DIR_PATH/f'{csv_path}/df_absolute_ohne_interpolation.feather')

df = fixing_faster_loop(df,csv_path,Save=True)


#csv_path_figures = 'Processing_Results/Figures/'
#Title ='DESO With Load Opitmization'

#df = pd.read_feather(FILE_DIR_PATH/f'SIL_Results/Actual/df_absolute.feather')


#plotting_virtual_HIL(df,csv_path_figures,Title=Title,Save =True,Show = False)

#print(df)