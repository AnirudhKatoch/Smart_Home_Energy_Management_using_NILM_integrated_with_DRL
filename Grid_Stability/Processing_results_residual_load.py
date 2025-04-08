from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

FILE_DIR_PATH = Path(__file__).parent

def alle(csv_path):

    def plotting_virtual_HIL(df,csv_path,Title,Save,Show):

        fig1, ax1 = plt.subplots(figsize=(6.4,4.8))

        df['Time_seconds'] = df.index/3600

        ax1.plot(df['Time_seconds'], df['Pmeas_kW_Grid']/1000, label='Grid AC Power',color ='red',linewidth=3)
        ax1.plot(df['Time_seconds'], df['Residual AC Power']/1000 , label='Residual AC Power', color='black')

        ax1.set_xlim(min(df['Time_seconds']), max(df['Time_seconds']))
        ax1.set_xlabel('Time (h)',fontsize=15)
        ax1.set_ylabel('Power (kW)',fontsize=15)

        if csv_path == 'Results/Actual/':
            ax1.legend(loc="lower left",fontsize=10)
        else:
            ax1.legend(loc="upper right", fontsize=10)

        ax1.tick_params(axis='x', labelsize=15)
        ax1.tick_params(axis='y', labelsize=15)
        ax1.grid(True)
        plt.tight_layout()
        #plt.xticks(range(25))


        if Save == True :
            fig1.savefig(FILE_DIR_PATH / f'{csv_path}/Power_Variation.png')

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

        df = pd.read_feather(FILE_DIR_PATH/f'{csv_path}/df_Grid_Stability.feather')

        df['Pmeas_kW_Grid'] = df['Pmeas_kW_Grid'] * 1000
        df['Vrms_meas_kV_Grid'] = df['Vrms_meas_kV_Grid'] * 1000

        df['time'] = pd.to_timedelta(df['time'])
        df['Time_seconds'] = df['time'].dt.total_seconds()

        df['Time_seconds'] = df['Time_seconds'] - min(df['Time_seconds'])
        df = df.drop(columns=['time'])
        df['Time_seconds'] = df['Time_seconds'].astype(int)

        df = df[df['Time_seconds'] < (Actual_Loop_time)] ################################################################ Check this value evertime to be sure that the results are correct.

        last_value_count = (df['Time_seconds'] == df['Time_seconds'].iloc[-1]).sum()
        if last_value_count < 10:
            df = df[df['Time_seconds'] != df['Time_seconds'].iloc[-1]]

        unique_time_seconds = df['Time_seconds'].unique()
        processed_dfs = []

        for time_sec in unique_time_seconds:
            #print(time_sec)
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
                #print(column)
                interpolation_function = CubicSpline(old_index, column)
                new_index = np.linspace(0, len(column) - 1, new_index_size)
                interpolated_values = interpolation_function(new_index)
                return (pd.DataFrame(interpolated_values, columns=[column.name]))

            df = pd.concat([interpolate_column(df[column]) for column in df], axis=1)

            return df

        df = Resolution_change(df,3600*24)
        df['Time_seconds'] = pd.Index(range(len(df)))

        rolling_means = df.drop(columns=["Time_seconds"]).rolling(window=25, min_periods=1).mean()
        rolling_means["Time_seconds"] = df["Time_seconds"]

        #rolling_means = rolling_means[:86400]

        if Save == True:
            rolling_means.to_feather(FILE_DIR_PATH / f'{csv_path}/df_absolute.feather')


        return rolling_means

    df = Faster_result_processing(FILE_DIR_PATH/csv_path, Actual_Loop_time = 96, Save = False)
    df = fixing_faster_loop(df,csv_path,Save=True)
    plotting_virtual_HIL(df,csv_path,Title='',Save =True,Show = False)


SCP = '0.2MVA'

csv_path = f'Voltages_Results/Average_day/{SCP}/Actual/'
alle(csv_path)

csv_path = f'Voltages_Results/Average_day/{SCP}/Optimized/'
alle(csv_path)

#########################################################################################################################

csv_path = f'Voltages_Results/Average_day/{SCP}/Actual/'
df_Actual = pd.read_feather(FILE_DIR_PATH/f'{csv_path}/df_absolute.feather')
Line_Voltage_Actual = df_Actual['Vrms_meas_kV_Grid']

csv_path = f'Voltages_Results/Average_day/{SCP}/Optimized/'
df_Optimized_Predicted = pd.read_feather(FILE_DIR_PATH/f'{csv_path}/df_absolute.feather')
Line_Voltage_Optimized_Predicted = df_Optimized_Predicted['Vrms_meas_kV_Grid']

Line_Voltage_Actual = Line_Voltage_Actual.rolling(window=250, min_periods=1).mean()
Line_Voltage_Optimized_Predicted = Line_Voltage_Optimized_Predicted.rolling(window=250, min_periods=1).mean()

fig1, ax1 = plt.subplots(figsize=(6.4,4.8))
df_Actual['Time_seconds'] = df_Actual.index/3600
df_Optimized_Predicted['Time_seconds'] = df_Optimized_Predicted.index/3600
ax1.plot(df_Actual['Time_seconds'], Line_Voltage_Actual , label='Scenario 1')
ax1.plot(df_Optimized_Predicted['Time_seconds'], Line_Voltage_Optimized_Predicted , label='Scenario 2')
ax1.axhline(y=400, color='red', linestyle='--')
ax1.set_xlim(min(df_Actual['Time_seconds']), max(df_Actual['Time_seconds']))
ax1.set_xlabel('Time (h)',fontsize=15)
ax1.set_ylabel('Line Voltage (V)',fontsize=15)
ax1.legend(loc="lower center",fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.grid(True)
plt.tight_layout()
fig1.savefig(FILE_DIR_PATH / f'Voltages_Results/Average_day/{SCP}/Grid_line_voltage_at_0.2_MVA.png')






