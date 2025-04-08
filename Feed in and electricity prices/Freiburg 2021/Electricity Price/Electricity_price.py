import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import timedelta


FILE_DIR_PATH = Path(__file__).parent


file_path = f"energy-charts_Electricity_production_and_spot_prices_in_Germany_in_2021.csv"
df = pd.read_csv(file_path, sep=';', parse_dates=['Date (GMT+1)'])
df.set_index('Date (GMT+1)', inplace=True)
df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Berlin')
df = df[~df.index.duplicated(keep='first')]
df.index.name = None
df.index = df.index.tz_convert('utc')
df.index = df.index + timedelta(hours=1)


df['price_EUR_kWh'] = df['Price (EUR/MWh, EUR/tCO2)'] / 1000
df['pay_price_EUR_kWh'] = (df['price_EUR_kWh'] + 0.15) * 1.19 # # Frederik's formula
df = df.drop(columns= ['Price (EUR/MWh, EUR/tCO2)', 'price_EUR_kWh','Nuclear  Power (MW)', 'Non-Renewable Power (MW)','Renewable Power (MW)' ])

df = df.resample('S').ffill()
new_end = pd.Timestamp("2021-12-31 23:59:59", tz="utc")
new_index = pd.date_range(start=df.index.min(), end=new_end, freq="S")
df = df.reindex(new_index)
df.fillna(0, inplace=True)

monthly_dfs = {}
def get_days_in_month(year, month):
    from calendar import monthrange
    return monthrange(year, month)[1]

year = df.index.year[0]

for month in range(1, 13):
    print(month)
    month_df = df[df.index.month == month]
    days_in_month = get_days_in_month(year, month)
    month_dict = {day: month_df[month_df.index.day == day] for day in range(1, days_in_month + 1)}
    month_dict = {day: list(month_dict[day].values.flatten()) for day in month_dict}
    month_df_from_dict = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in month_dict.items()]))
    month_df_from_dict.columns = [f'{month:02d} Day {day}' for day in month_df_from_dict.columns]
    monthly_dfs[month] = month_df_from_dict

full_year_df = pd.concat(monthly_dfs.values(), axis=1)

# Calculate row averages
row_averages = full_year_df.mean(axis=1)

# Calculate standard deviation for each row
row_std_devs = full_year_df.std(axis=1)

# Create the final DataFrame with averages and standard deviations
average_df = pd.DataFrame({
    'Electricity_Price_mean': row_averages,
    'Electricity_Price_max_1': row_averages + row_std_devs,
    'Electricity_Price_min_1': row_averages - row_std_devs,
    'Electricity_Price_max_2': row_averages + 2 * row_std_devs,
    'Electricity_Price_min_2': row_averages - 2 * row_std_devs,
    'Electricity_Price_max_3': row_averages + 3 * row_std_devs,
    'Electricity_Price_min_3': row_averages - 3 * row_std_devs
})
average_df = average_df.clip(lower=0)

Time_period = average_df.index
Time_period = Time_period/3600

# Plotting the first figure
plt.figure(figsize=(10, 6))
plt.plot(Time_period, average_df['Electricity_Price_mean'], label='Electricity_Price Mean', color='blue')
plt.fill_between(Time_period, average_df['Electricity_Price_min_1'], average_df['Electricity_Price_max_1'], color='orange', alpha=0.3,
                 label='±1σ Range')
plt.fill_between(Time_period, average_df['Electricity_Price_min_2'], average_df['Electricity_Price_max_2'], color='green', alpha=0.2,
                 label='±2σ Range')
plt.fill_between(Time_period, average_df['Electricity_Price_min_3'], average_df['Electricity_Price_max_3'], color='green', alpha=0.1,
                 label='±3σ Range')

plt.xlabel('Time (Hours)')
plt.ylabel('Electricity Price (EUR/kWh)')
plt.title('Electricity Price Mean and Standard Deviation Ranges for year 2021 ')
plt.xlim(min(Time_period),max(Time_period))
plt.legend()
plt.grid(True)

plt.savefig(FILE_DIR_PATH / f'Electricity_Price.png')

average_df.to_csv(FILE_DIR_PATH / f'Electricity_Price.csv',sep=';')

plt.show()




'''
# Plotting the second figure
plt.figure(figsize=(14, 7))
plt.plot(Time_period, average_df['Electricity_Price_mean'], label='Electricity_Price Mean', color='blue')
plt.plot(Time_period, average_df['Electricity_Price_max_1'], label='Electricity_Price Mean + 1σ', color='orange')
plt.plot(Time_period, average_df['Electricity_Price_min_1'], label='Electricity_Price Mean - 1σ', color='orange')
plt.plot(Time_period, average_df['Electricity_Price_max_2'], label='Electricity_Price Mean + 2σ', color='red')
plt.plot(Time_period, average_df['Electricity_Price_min_2'], label='Electricity_Price Mean - 2σ', color='red')
plt.plot(Time_period, average_df['Electricity_Price_max_3'], label='Electricity_Price Mean + 3σ', color='green')
plt.plot(Time_period, average_df['Electricity_Price_min_3'], label='Electricity_Price Mean - 3σ', color='green')

plt.xlabel('Time')
plt.ylabel('Load')
plt.title('PV  Mean and Standard Deviation Ranges for 2021 (Lines)')
plt.legend()
plt.grid(True)

#plt.savefig(FILE_DIR_PATH / f'Freiburg PV Profile/Freiburg_PV_Profile_Lines.png')
'''

