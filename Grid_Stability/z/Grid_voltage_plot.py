from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

FILE_DIR_PATH = Path(__file__).parent


csv_path = f'Results/Actual/'
df_Actual = pd.read_feather(FILE_DIR_PATH/f'{csv_path}/df_absolute.feather')
Line_Voltage_Actual = df_Actual['Three-phase Meter GRID.VAB_RMS']

csv_path = f'Results/Optimized_Predicted/'
df_Optimized_Predicted = pd.read_feather(FILE_DIR_PATH/f'{csv_path}/df_absolute.feather')
Line_Voltage_Optimized_Predicted = df_Optimized_Predicted['Three-phase Meter GRID.VAB_RMS']


Line_Voltage_Actual = Line_Voltage_Actual.rolling(window=250, min_periods=1).mean()
#Line_Voltage_Optimized_Predicted = Line_Voltage_Optimized_Predicted.rolling(window=250, min_periods=1).mean()



fig1, ax1 = plt.subplots(figsize=(6.4,4.8))
df_Actual['Time_seconds'] = df_Actual.index/3600
#ax1.plot(df_Actual['Time_seconds'], Line_Voltage_Actual , label='Scenario 1')
ax1.plot(df_Optimized_Predicted['Time_seconds'], Line_Voltage_Optimized_Predicted , label='Scenario 2')
ax1.set_xlim(min(df_Actual['Time_seconds']), max(df_Actual['Time_seconds']))
ax1.set_xlabel('Time (h)',fontsize=15)
ax1.set_ylabel('Line Voltage (V)',fontsize=15)
ax1.legend(loc="lower left",fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.grid(True)
plt.tight_layout()
fig1.savefig(FILE_DIR_PATH / f'Figures/Line_Voltage.png')


fig2, ax2 = plt.subplots(figsize=(6.4,4.8))
df_Actual['Time_seconds'] = df_Actual.index/3600
ax2.plot(df_Actual['Time_seconds'], df_Actual['Three-phase Meter GRID.VAn_RMS']  , label='Scenario 1')
ax2.plot(df_Optimized_Predicted['Time_seconds'], df_Optimized_Predicted['Three-phase Meter GRID.VAn_RMS']  , label='Scenario 2')
ax2.set_xlim(min(df_Actual['Time_seconds']), max(df_Actual['Time_seconds']))
ax2.set_xlabel('Time (h)', fontsize=15)
ax2.set_ylabel('Line Voltage (V)', fontsize=15)
ax2.legend(loc="lower left",fontsize=15)
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
ax2.grid(True)
plt.tight_layout()
#fig2.savefig(FILE_DIR_PATH / f'Figures/Phase_Voltage.png')