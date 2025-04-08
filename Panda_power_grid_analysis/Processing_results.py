from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCP = 5

df_Actual = pd.read_csv(f'res_line/loading_percent_Residual_Actual_{SCP}.csv',sep=';')
df_Actual = df_Actual.drop(columns=['Unnamed: 0'])
Time = df_Actual.index/4

df_Optimized = pd.read_csv(f'res_line/loading_percent_Residual_Optimized_{SCP}.csv',sep=';')
df_Optimized = df_Optimized.drop(columns=['Unnamed: 0'])

plt.figure(figsize=(6.4, 4.8))  # Set the figure size (optional)
plt.plot(Time, df_Actual.iloc[:, :-1].mean(axis=1), label='Scenario 1')
plt.plot(Time, df_Optimized.iloc[:, :-1].mean(axis=1), label='Scenario 2')
# Add labels and title
plt.xlabel('Time (h)', fontsize=15)
plt.ylabel('Line Loading (%)', fontsize=15)
plt.xlim(min(Time),max(Time))
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.grid()
plt.tight_layout()
plt.legend(fontsize=15)
plt.savefig(f'Figures/Average line Loading at {SCP} MVA.svg')

df_Actual = pd.read_csv(f'res_bus/vm_pu_Residual_Actual_{SCP}.csv',sep=';')
df_Actual = df_Actual.drop(columns=['Unnamed: 0'])
Time = df_Actual.index/4

df_Optimized = pd.read_csv(f'res_bus/vm_pu_Residual_Optimized_{SCP}.csv',sep=';')
df_Optimized = df_Optimized.drop(columns=['Unnamed: 0'])

plt.figure(figsize=(6.4, 4.8))
plt.plot(Time, df_Actual.iloc[:, :-2].mean(axis=1)*400, label='Scenario 1')
plt.plot(Time, df_Optimized.iloc[:, :-2].mean(axis=1)*400, label='Scenario 2')
plt.xlabel('Time (h)', fontsize=15)
plt.ylabel('Voltage (V)', fontsize=15)
plt.xlim(min(Time),max(Time))
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.grid()
plt.tight_layout()
plt.legend(fontsize=15)
plt.savefig(f'Figures/Average voltage deviation at {SCP} MVA.svg')