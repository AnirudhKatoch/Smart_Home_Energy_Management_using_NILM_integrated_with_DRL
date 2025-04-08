from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FILE_DIR_PATH = Path(__file__).parent.parent

df = pd.read_csv(FILE_DIR_PATH/f'Grid_Stability/Database/Average_day/5MVA/Residual_load.csv',sep=';')

print(df)
plt.plot(df['Residual_Actual'])
plt.plot(df['Residual_Optimized'])


df_actual = pd.DataFrame({
    'Load_0': df['Residual_Actual'],
    'Load_1': df['Residual_Actual'],
    'Load_2': df['Residual_Actual'],
    'Load_3': df['Residual_Actual'],
    'Load_4': df['Residual_Actual'],
    'Load_5': df['Residual_Actual'],
    'Load_6': df['Residual_Actual'],
    'Load_7': df['Residual_Actual'],
    'Load_8': df['Residual_Actual'],
    'Load_9': df['Residual_Actual'],
    'Load_10': df['Residual_Actual'],
    'Load_11': df['Residual_Actual'],
    'Load_12': df['Residual_Actual']
})

#df_actual.columns = [None] * len(df_actual.columns)
df_actual.to_csv(FILE_DIR_PATH/f'Panda_power_grid_analysis/Database/Residual_Actual.csv',sep=';')

df_optimized = pd.DataFrame({
    'Load_0': df['Residual_Optimized'],
    'Load_1': df['Residual_Optimized'],
    'Load_2': df['Residual_Optimized'],
    'Load_3': df['Residual_Optimized'],
    'Load_4': df['Residual_Optimized'],
    'Load_5': df['Residual_Optimized'],
    'Load_6': df['Residual_Optimized'],
    'Load_7': df['Residual_Optimized'],
    'Load_8': df['Residual_Optimized'],
    'Load_9': df['Residual_Optimized'],
    'Load_10': df['Residual_Optimized'],
    'Load_11': df['Residual_Optimized'],
    'Load_12': df['Residual_Optimized']
})

df_optimized.to_csv(FILE_DIR_PATH/f'Panda_power_grid_analysis/Database/Residual_Optimized.csv',sep=';')