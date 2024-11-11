import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FILE_DIR_PATH = Path(__file__).parent.parent

start_date = f'2021-01-01'
end_date = f'2021-01-01 23:59'
date_range = pd.date_range(start=start_date, end=end_date, freq='T')

df = pd.DataFrame(index=date_range)
pattern = [0] * 30 + [75] * 30
refrigerator_values = pattern * (len(df) // len(pattern))
remaining = 525600 % len(pattern)
if remaining > 0:
    refrigerator_values += pattern[:remaining]

df['Occupied'] = True
df['Refrigerator'] = refrigerator_values


for day in pd.date_range('2021-01-01', '2021-12-31'):

    start_time = f"{day.strftime('%Y-%m-%d')} 23:30:00"
    end_time = f"{day.strftime('%Y-%m-%d')} 23:59:00"
    df.loc[start_time:end_time, 'Refrigerator'] = 0


differences = (df['Refrigerator'].diff())
differences.fillna(0, inplace=True)

extended_indices = []

Refrigerator_start = differences[differences == 75].index
Refrigerator_start = pd.DatetimeIndex(sorted(set(Refrigerator_start - pd.Timedelta(minutes=1)) | set(Refrigerator_start) | set(Refrigerator_start + pd.Timedelta(minutes=1))))

Refrigerator_end = differences[differences == -75].index
Refrigerator_end = pd.DatetimeIndex(sorted(set(Refrigerator_end - pd.Timedelta(minutes=2)) | set(Refrigerator_end - pd.Timedelta(minutes=1)) | set(Refrigerator_end)))


df.loc[Refrigerator_start, 'Occupied'] = False
df.loc[Refrigerator_end, 'Occupied'] = False



def Load_engine(df, column_name, power, duration):

    df[column_name] = 0

    # Loop over each day
    for day in pd.date_range(start=start_date, end=end_date, freq='D'):

        start_of_day = day
        valid_start = False

        # Keep generating start times until a valid one is found
        while not valid_start:
            random_minute = np.random.randint(0, 1440 - duration)
            start_time = start_of_day + pd.Timedelta(minutes=random_minute)
            end_time = start_time + pd.Timedelta(minutes=duration - 1)

            # Check if both start_time and end_time are 'Occupied'

            if df.at[start_time, 'Occupied'] and df.at[end_time, 'Occupied']:
                valid_start = True

        df.loc[start_time + pd.Timedelta(minutes=1), 'Occupied'] = False
        df.loc[start_time, 'Occupied'] = False
        df.loc[start_time - pd.Timedelta(minutes=1), 'Occupied'] = False

        df.loc[end_time + pd.Timedelta(minutes=1), 'Occupied'] = False
        df.loc[end_time, 'Occupied'] = False
        df.loc[end_time - pd.Timedelta(minutes=1), 'Occupied'] = False

        df.loc[start_time:end_time, column_name] = power

    return df


df = Load_engine(df, power=80, duration=250, column_name='Boiler')
df = Load_engine(df, power=500, duration=67, column_name='Wash_Dryer')
df = Load_engine(df, power=1200, duration=120, column_name='Dish_Washer')
df = Load_engine(df, power=55, duration=200, column_name='Solar_Thermal_Pumping_Station')
df = Load_engine(df, power=32, duration=150, column_name='Laptop_Computer')
df = Load_engine(df, power=100, duration=175, column_name='Television')
df = Load_engine(df, power=65, duration=325, column_name='Home_Theater_PC')
df = Load_engine(df, power=1900, duration=10, column_name='Kettle')
df = Load_engine(df, power=1000, duration=15, column_name='Toaster')
df = Load_engine(df, power=400, duration=20, column_name='Microwave')
df = Load_engine(df, power=40, duration=180, column_name='Computer_Monitor')
df = Load_engine(df, power=15, duration=95, column_name='Audio_System')
df = Load_engine(df, power=1750, duration=45, column_name='Vaccum_Cleaner')
df = Load_engine(df, power=350, duration=7, column_name='Hair_Dryer')
df = Load_engine(df, power=60, duration=7, column_name='Hair_Straighteners')
df = Load_engine(df, power=1100, duration=8, column_name='Clothing_Iron')
df = Load_engine(df, power=950, duration=60, column_name='Oven')
df = Load_engine(df, power=25, duration=100, column_name='Fan')
df = Load_engine(df, power=450, duration=100, column_name='Washing_Machine')
df = Load_engine(df, power=2000, duration=60, column_name='EV')
df = Load_engine(df, power=1125, duration=30, column_name='Electric_Water_Heater')
df = Load_engine(df, power=460, duration=65, column_name='Washing_Machine_2')

df = df.drop(columns = ['Occupied'])
df['Total_Power'] = df.sum(axis=1)

df.to_csv( FILE_DIR_PATH / f'Databases/Load_Profile_for_HIL.csv', sep=';' )

df = df.reset_index()
#df = df[:60*24]
fig1,ax1 = plt.subplots(figsize=(20, 7))
ax1.plot(df.index/60, df['Boiler'],label = 'Boiler')
ax1.plot(df.index/60, df['Wash_Dryer'],label = 'Wash_Dryer')
ax1.plot(df.index/60, df['Dish_Washer'],label = 'Dish_Washer')
ax1.plot(df.index/60, df['Solar_Thermal_Pumping_Station'],label = 'Solar_Thermal_Pumping_Station')
ax1.plot(df.index/60, df['Laptop_Computer'],label = 'Laptop_Computer')
ax1.plot(df.index/60, df['Television'],label = 'Television')
ax1.plot(df.index/60, df['Home_Theater_PC'],label = 'Home_Theater_PC')
ax1.plot(df.index/60, df['Kettle'],label = 'Kettle')
ax1.plot(df.index/60, df['Toaster'],label = 'Toaster')
ax1.plot(df.index/60, df['Hair_Dryer'],label = 'Hair_Dryer')
ax1.plot(df.index/60, df['Hair_Straighteners'],label = 'Hair_Straighteners')
ax1.plot(df.index/60, df['Clothing_Iron'],label = 'Clothing_Iron')
ax1.plot(df.index/60, df['Oven'],label = 'Oven')
ax1.plot(df.index/60, df['Fan'],label = 'Fan')
ax1.plot(df.index/60, df['Washing_Machine'],label = 'Washing_Machine')
ax1.plot(df.index/60, df['EV'],label = 'EV')
ax1.plot(df.index/60, df['Refrigerator'],label = 'Refrigerator')
ax1.plot(df.index/60, df['Electric_Water_Heater'],label = 'Electric_Water_Heater')
ax1.plot(df.index/60, df['Washing_Machine_2'],label = 'Washing_Machine_2')
ax1.set_xlim(min(df.index/60),max(df.index/60))
#plt.ylim(0,max(df['Total_Power'])+10)
#ax1.set_ylim(0,200)
ax1.set_xlabel('Time (Hours)')
ax1.set_ylabel('Power (W)')
ax1.set_title('Load for a 24 hour period (Predicted)')
plt.grid()
plt.legend()
fig1.savefig(FILE_DIR_PATH/f'Figures/Load_Profile_for_HIL.png')
#plt.show()















