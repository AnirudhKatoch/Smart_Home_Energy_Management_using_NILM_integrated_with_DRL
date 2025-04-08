from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_DIR_PATH = Path(__file__).parent

df = pd.read_csv(FILE_DIR_PATH/f'Database/SIL_year_inputs_data.csv',sep=';')
df = df.drop(columns=['Unnamed: 0', 'PV_Irradiance','Time_unix'])

df['Load_actual_kW'] = df['Load_actual_kW'] * 1000
df['Load_Optimized_predicted_kW'] = df['Load_Optimized_predicted_kW'] * 1000


df = df[:96*20]

window_size = 96
num_windows = len(df) // 96

windows_actual = []
windows_predicted = []

for i in range(num_windows):
    start_idx = i * window_size
    end_idx = start_idx + window_size
    windows_actual.append(df['Load_actual_kW'].iloc[start_idx:end_idx].reset_index(drop=True))
    windows_predicted.append(df['Load_Optimized_predicted_kW'].iloc[start_idx:end_idx].reset_index(drop=True))

stacked_actual = pd.concat(windows_actual, axis=1)
stacked_predicted = pd.concat(windows_predicted, axis=1)

sum_actual = stacked_actual.sum(axis=1)
sum_predicted = stacked_predicted.sum(axis=1)

# Convert the results to a DataFrame
result_df = pd.DataFrame({
    "Load_actual_W": sum_actual,
    "Optimized_predicted_W": sum_predicted
})

print(result_df)

result_df.to_csv(FILE_DIR_PATH/f'Database/Load_input.csv',sep=';')

