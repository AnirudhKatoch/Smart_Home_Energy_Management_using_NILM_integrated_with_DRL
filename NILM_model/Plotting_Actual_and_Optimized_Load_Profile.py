import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

FILE_DIR_PATH = Path(__file__).parent


df = pd.read_csv(FILE_DIR_PATH/f'Databases/Load_Profile_k_NN_model_learn.csv',sep=';')

df = df.reset_index()

df= df[:1440]


#df = df[:60*24]
fig1,ax1 = plt.subplots(figsize=(6.4, 4.8))

ax1.plot(df.index/60, df['Refrigerator'],label = 'Refrigerator')
ax1.plot(df.index/60, df['Boiler'],label = 'Boiler')
ax1.plot(df.index/60, df['Wash_Dryer'],label = 'Wash Dryer')
ax1.plot(df.index/60, df['Dish_Washer'],label = 'Dish Washer')
ax1.plot(df.index/60, df['Solar_Thermal_Pumping_Station'],label = 'Pumping Station')
ax1.plot(df.index/60, df['Laptop_Computer'],label = 'Laptop Computer')
ax1.plot(df.index/60, df['Television'],label = 'Television')
ax1.plot(df.index/60, df['Home_Theater_PC'],label = 'Home Theater PC')
ax1.plot(df.index/60, df['Kettle'],label = 'Kettle')
ax1.plot(df.index/60, df['Toaster'],label = 'Toaster')
ax1.plot(df.index/60, df['Microwave'],label = 'Microwave')
ax1.plot(df.index/60, df['Computer_Monitor'],label = 'Computer Monitor')
ax1.plot(df.index/60, df['Audio_System'],label = 'Audio System')
ax1.plot(df.index/60, df['Vaccum_Cleaner'],label = 'Vaccum Cleaner')
ax1.plot(df.index/60, df['Hair_Dryer'],label = 'Hair Dryer')
ax1.plot(df.index/60, df['Hair_Straighteners'],label = 'Hair Straighteners')
ax1.plot(df.index/60, df['Clothing_Iron'],label = 'Clothing Iron')
ax1.plot(df.index/60, df['Oven'],label = 'Oven')
ax1.plot(df.index/60, df['Fan'],label = 'Fan')
ax1.plot(df.index/60, df['Washing_Machine'],label = 'Washing Machine')
ax1.plot(df.index/60, df['EV'],label = 'EV')
ax1.plot(df.index/60, df['Electric_Water_Heater'],label = 'Water Heater')
ax1.plot(df.index/60, df['Washing_Machine_2'],label = 'Washing Machine 2')


#ax1.plot(df.index/60, df['Total_Power'],label = 'Total_Power')
ax1.set_xlim(min(df.index/60),max(df.index/60))
#plt.ylim(0,max(df['Total_Power'])+10)
#ax1.set_ylim(0,200)
ax1.set_xlabel('Time (h)', fontsize=15)
ax1.set_ylabel('Power (W)', fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
#ax1.set_title('Custom Load profile for a period of 24 hour')
#plt.tight_layout()
#plt.subplots_adjust(left=0.1,right=0.70, top=0.90)
#plt.grid()
plt.tight_layout()
#ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=15)
fig1.savefig(FILE_DIR_PATH/f'Figures/Disaggregated household’s total power consumption.png')
#plt.show()

#df = df[:60*24]

fig2,ax2 = plt.subplots(figsize=(6.4, 4.8))
ax2.plot(df.index / 60, df['Total_Power'], label='Total Power')
ax2.set_xlim(min(df.index / 60), max(df.index / 60))
ax2.set_xlabel('Time (h)', fontsize=15)
ax2.set_ylabel('Power (W)', fontsize=15)
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
#plt.subplots_adjust(left=0.1,right=0.70, top=0.99)
#plt.grid()
plt.tight_layout()
#ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=15)
fig2.savefig(FILE_DIR_PATH / 'Figures/Household’s total power consumption.png' )

#plt.show()
