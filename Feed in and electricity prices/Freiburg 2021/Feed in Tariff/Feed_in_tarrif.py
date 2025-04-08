import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

FILE_DIR_PATH = Path(__file__).parent

file_path = 'Feed-in-Tarrif 2021 Germany Rofftop 10 kWp.xlsx'

df_original = pd.read_excel(file_path,sheet_name='Feed in Tarrif 2021 Germany')
month_to_value = dict(zip(df_original['Month '], df_original['bis 10 kW']))

date_range = pd.date_range(start='2021-01-01', end='2021-12-31 23:59:59', freq='S')
df = pd.DataFrame(index=date_range)
df['Feed_in_Tariff_EUR_kWh'] = None

for month, value in month_to_value.items():
    df.loc[df.index.month == month, 'Feed_in_Tariff_EUR_kWh'] = value

df['Feed_in_Tariff_EUR_kWh']    = df['Feed_in_Tariff_EUR_kWh'] / 100

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
    'Tariff_mean': row_averages,
    'Tariff_max_1': row_averages + row_std_devs,
    'Tariff_min_1': row_averages - row_std_devs,
    'Tariff_max_2': row_averages + 2 * row_std_devs,
    'Tariff_min_2': row_averages - 2 * row_std_devs,
    'Tariff_max_3': row_averages + 3 * row_std_devs,
    'Tariff_min_3': row_averages - 3 * row_std_devs
})
average_df = average_df.clip(lower=0)

print(average_df)

Time_period = average_df.index
Time_period = Time_period/3600

# Plotting the first figure
plt.figure(figsize=(14, 7))
plt.plot(Time_period, average_df['Tariff_mean'], label='Tariff Mean', color='blue')
plt.fill_between(Time_period, average_df['Tariff_min_1'], average_df['Tariff_max_1'], color='orange', alpha=0.3,
                 label='±1σ Range')
plt.fill_between(Time_period, average_df['Tariff_min_2'], average_df['Tariff_max_2'], color='green', alpha=0.2,
                 label='±2σ Range')
plt.fill_between(Time_period, average_df['Tariff_min_3'], average_df['Tariff_max_3'], color='green', alpha=0.1,
                 label='±3σ Range')

plt.xlabel('Time (Hours)')
plt.ylabel('Feed in Tariff (EUR/kWh)')
plt.title('Feed in Tariff Mean and Standard Deviation Ranges for year 2021 for 10 kWp ')
plt.xlim(min(Time_period),max(Time_period))
plt.legend()
plt.grid(True)

plt.savefig(FILE_DIR_PATH / f'Feed_in_Tariff_EUR_kWh.png')

average_df.to_csv(FILE_DIR_PATH / f'Feed_in_Tariff_EUR_kWh.csv',sep=';')

plt.show()



'''
# Plotting the second figure
plt.figure(figsize=(14, 7))
plt.plot(Time_period, average_df['Tariff_mean'], label='Tariff Mean', color='blue')
plt.plot(Time_period, average_df['Tariff_max_1'], label='Tariff Mean + 1σ', color='orange')
plt.plot(Time_period, average_df['Tariff_min_1'], label='Tariff Mean - 1σ', color='orange')
plt.plot(Time_period, average_df['Tariff_max_2'], label='Tariff Mean + 2σ', color='red')
plt.plot(Time_period, average_df['Tariff_min_2'], label='Tariff Mean - 2σ', color='red')
plt.plot(Time_period, average_df['Tariff_max_3'], label='Tariff Mean + 3σ', color='green')
plt.plot(Time_period, average_df['Tariff_min_3'], label='Tariff Mean - 3σ', color='green')

plt.xlabel('Time')
plt.ylabel('Load')
plt.title('PV  Mean and Standard Deviation Ranges for 2021 (Lines)')
plt.legend()
plt.grid(True)

#plt.savefig(FILE_DIR_PATH / f'Freiburg PV Profile/Freiburg_PV_Profile_Lines.png')
'''

