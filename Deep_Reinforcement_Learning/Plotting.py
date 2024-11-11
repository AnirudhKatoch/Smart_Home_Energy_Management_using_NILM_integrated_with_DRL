import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FILE_DIR_PATH = Path(__file__).parent.parent

'''

df = pd.read_csv(FILE_DIR_PATH/'Database/PV_elec/1_min_resolution/DRL_1_day_mean_inputs.csv',sep=';')
df_action = pd.read_csv(FILE_DIR_PATH/'Trial_1/Results/Best_action_PPO.csv',sep=';')

df['Washing_Machine'] = 0
df.loc[int(df_action['Washing_Machine'][0] * 60) : int(df_action['Washing_Machine'][0] * 60 + 90), 'Washing_Machine'] = 1000
#df.loc[int(17 * 60) : int(17 * 60 + 90), 'Washing_Machine'] = 1000

df['Cloth_Dryer'] = 0
df.loc[int(df_action['Cloth_Dryer'][0] * 60) : int(df_action['Cloth_Dryer'][0] * 60 + 60), 'Cloth_Dryer'] = 3000
#df.loc[int(18.50 * 60) : int(18.50 * 60 + 60), 'Cloth_Dryer'] = 3000

df['Dish_Washer'] = 0
df.loc[int(df_action['Dish_Washer'][0] * 60) : int(df_action['Dish_Washer'][0] * 60 + 120), 'Dish_Washer'] = 2000
#df.loc[int(21 * 60) : int(21 * 60 + 120), 'Dish_Washer'] = 2000

df['EV'] = 0
df.loc[int(df_action['EV'][0] * 60) : int(df_action['EV'][0] * 60 + 240), 'EV'] = 5000
#df.loc[int(17 * 60) : int(17 * 60 + 240), 'EV'] = 5000

df['Electric_Water_Heater'] = 0
df.loc[int(df_action['Electric_Water_Heater'][0] * 60) : int(df_action['Electric_Water_Heater'][0] * 60 + 90), 'Electric_Water_Heater'] = 4000
#df.loc[int(6 * 60) : int(6 * 60 + 120), 'Electric_Water_Heater'] = 4000

df['Load'] = df['Washing_Machine'] + df['Cloth_Dryer'] + df['Dish_Washer'] + df['EV'] + df['Electric_Water_Heater']
df['Grid'] = df['Load'] - df['PV_Power_W']


fig, ax3 = plt.subplots(figsize=(6.4, 4.8))
ax3.plot( df.index/60, df['PV_Power_W'], label = 'PV Power')
#ax3.plot( df.index/60, df['Washing_Machine'], label = 'Washing\n Machine')
#ax3.plot( df.index/60, df['Cloth_Dryer'], label = 'Cloth\n Dryer')
#ax3.plot( df.index/60, df['Dish_Washer'], label = 'Dish\n Washer')
#ax3.plot( df.index/60, df['EV'], label = 'EV')
#ax3.plot( df.index/60, df['Electric_Water_Heater'], label = 'Electric Water\n Heater')
ax3.plot( df.index/60, df['Load'], label = 'Load')
ax3.plot( df.index/60, df['Grid'], label = 'Grid')
ax3.set_xlabel('Time (h)', fontsize=15)
ax3.set_ylabel('Power (W)', fontsize=15)
ax3.tick_params(axis='x', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)
plt.xlim(min(df.index/60),max(df.index/60))
ax3.grid()
ax3.legend(loc='upper left',fontsize=15)
plt.tight_layout()
#plt.subplots_adjust(left=0.175, top=0.99)
plt.savefig(FILE_DIR_PATH/f'Trial_1/figures/Appliance.png')
'''
'''

df['Washing_Machine'] = 0
df.loc[int(17 * 60) : int(17 * 60 + 90), 'Washing_Machine'] = 1000

df['Cloth_Dryer'] = 0
df.loc[int(19 * 60) : int(19 * 60 + 60), 'Cloth_Dryer'] = 3000

df['Dish_Washer'] = 0
df.loc[int(21 * 60) : int(21 * 60 + 120), 'Dish_Washer'] = 2000

df['EV'] = 0
df.loc[int(18 * 60) : int(18 * 60 + 240), 'EV'] = 5000

df['Electric_Water_Heater'] = 0
df.loc[int(6 * 60) : int(6 * 60 + 90), 'Electric_Water_Heater'] = 4000


df['Load'] = df['Washing_Machine'] + df['Cloth_Dryer'] + df['Dish_Washer'] + df['EV'] + df['Electric_Water_Heater']

#df['Grid'] = df['Load'] - df['PV_Power_W']



fig, ax4 = plt.subplots(figsize=(10,6))
#ax4.plot( df.index/60, df['PV_Power_W'], label = 'PV Power')
#ax4.plot( df.index/60, df['Grid'], label = 'Grid')
ax4.plot( df.index/60, df['Washing_Machine'], label = 'Washing Machine')
ax4.plot( df.index/60, df['Cloth_Dryer'], label = 'Cloth Dryer')
ax4.plot( df.index/60, df['Dish_Washer'], label = 'Dish Washer')
ax4.plot( df.index/60, df['EV'], label = 'EV')
ax4.plot( df.index/60, df['Electric_Water_Heater'], label = 'Electric Water Heater')
#ax4.plot( df.index/60, df['Load'], label = 'Load')

ax4.set_xlabel('Time (h)')
ax4.set_ylabel('Power (W)')
ax4.set_title('Appliances Individual Load')
plt.xlim(min(df.index/60),max(df.index/60))
ax4.grid()
ax4.legend()
plt.savefig(FILE_DIR_PATH/f'Trial_1/figures/Appliance_original.png')

#df['Grid_Feed_in'] = df['Grid'].apply(lambda x: x if x < 0 else 0)
#df['Grid_Supply'] = df['Grid'].apply(lambda x: x if x > 0 else 0)

df['Grid_Feed_in'] = df['Grid_Feed_in'] * (1/60) * -1 * df['Feed_in_Tariff_EUR_Wh']
df['Grid_Supply'] = df['Grid_Supply'] * (1/60) * df['Electricity_Price_EUR_Wh']

Total = df['Grid_Feed_in']  - df['Grid_Supply']
print(Total.sum())
'''

'''


df = pd.read_csv(FILE_DIR_PATH/'Database/PV_elec/1_min_resolution/DRL_1_day_mean_inputs.csv',sep=';')

fig, ax1 = plt.subplots(figsize=(6.4,4.8))
ax1.plot( df.index/60, df['PV_Power_W'], label = 'PV Power')
ax1.set_xlabel('Time (h)', fontsize=15)
ax1.set_ylabel('Power (W)', fontsize=15)
#ax1.set_title('PV Power over a period of 24 hours')
ax1.tick_params(axis='both', labelsize=15)  # Increase the font size of the numbers
plt.xlim(min(df.index/60),max(df.index/60))
ax1.grid()
ax1.legend(fontsize=15)
plt.tight_layout()
plt.savefig(FILE_DIR_PATH/f'Trial_1/figures/PV Power forecast.png')

fig, ax2 = plt.subplots(figsize=(6.4,4.8))
ax2.plot( df.index/60, df['Electricity_Price_EUR_Wh']*1000, label = 'Supply Price')
ax2.plot( df.index/60, df['Feed_in_Tariff_EUR_Wh']*1000, label = 'Feed in Price')
ax2.set_xlabel('Time (h)', fontsize=15)
ax2.set_ylabel('Price (EUR/kWh)', fontsize=15)
#ax2.set_title('Electricity Price over a period of 24 hours')
ax2.tick_params(axis='both', labelsize=15)  # Increase the font size of the numbers
plt.xlim(min(df.index/60),max(df.index/60))
ax2.grid()
ax2.legend(fontsize=15)
plt.tight_layout()
plt.savefig(FILE_DIR_PATH/f'Trial_1/figures/Supply and Feed in Price.png')

'''

df_PPO = pd.read_csv(FILE_DIR_PATH/'Trial_1/Results/episode_rewards_all_appliances_PPO.csv',sep=';')
df_A2C = pd.read_csv(FILE_DIR_PATH/'Trial_1/Results/episodes_rewards_all_appliances_A2C.csv',sep=';')

fig, ax5 = plt.subplots(figsize=(6.4,4.8))
ax5.plot(df_PPO.index[:150000], df_PPO['episode_reward'][:150000], label='PPO')
#ax5.plot(df_A2C.index[:150000], df_A2C['episode_reward'][:150000], label='A2C')
# Set the labels with increased font size
ax5.set_xlabel('Episodes', fontsize=15)
ax5.set_ylabel('Profit (Euro)', fontsize=15)
plt.xlim(min(df_PPO.index[:150000]), max(df_PPO.index[:150000]))
ax5.set_xticks([50000, 100000, 150000])
ax5.tick_params(axis='x', labelsize=15)
ax5.tick_params(axis='y', labelsize=15)
#plt.subplots_adjust(top=0.99)
plt.tight_layout()
#plt.legend(fontsize=15)
#ax5.grid()
plt.savefig(FILE_DIR_PATH/f'Trial_1/figures/DRL model training using PPO Policy.png')


