import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

FILE_DIR_PATH = Path(__file__).parent

Load_Profile = 'Load_Profile_for_HIL'

df_actual = pd.read_csv(FILE_DIR_PATH/f'Databases/{Load_Profile}.csv',sep=';')

df_actual = df_actual.loc[:, ~df_actual.columns.str.contains('^Unnamed')]

#df_actual = df_actual.drop(column = ['Unnamed: 0'] )

df_actual.to_csv(FILE_DIR_PATH/f'Results/{Load_Profile}/Actual.csv',sep=';')

df_actual = df_actual#[:60*24]

differences = df_actual['Total_Power'].diff()
differences.fillna(0, inplace=True)
differences = differences.values.reshape(-1, 1)

loaded_knn = joblib.load('k_NN_model/knn_model.pkl')
predictions = loaded_knn.predict(differences)


def Predictions_calculator(predictions, On, Off, Power):

    Predicted_array = np.where((predictions == On) | (predictions == Off), predictions, 0)

    indices_On = np.where(Predicted_array == On)[0]
    indices_Off = np.where(Predicted_array == Off)[0]

    for start in indices_On:
        next_Off = indices_Off[indices_Off > start]
        if len(next_Off) == 0:
            break
        end = next_Off[0]
        for i in range(start, end):
            Predicted_array[i] = Power
            Predicted_array[end] = 0

    return Predicted_array

df = pd.DataFrame()

df['Boiler'] =                        Predictions_calculator(predictions = predictions,On = 1 ,Off = 2 ,Power = 80 )
df['Wash_Dryer'] =                    Predictions_calculator(predictions = predictions,On = 3 ,Off = 4 ,Power = 500 )
df['Dish_Washer'] =                   Predictions_calculator(predictions = predictions,On = 5 ,Off = 6 ,Power = 1200 )
df['Solar_Thermal_Pumping_Station'] = Predictions_calculator(predictions = predictions,On = 7 ,Off = 8 ,Power = 55 )
df['Laptop_Computer'] =               Predictions_calculator(predictions = predictions,On = 9 ,Off = 10 ,Power = 32 )
df['Television'] =                    Predictions_calculator(predictions = predictions,On = 11 ,Off = 12 ,Power = 100 )
df['Home_Theater_PC'] =               Predictions_calculator(predictions = predictions,On = 13 ,Off = 14 ,Power = 65 )
df['Kettle'] =                        Predictions_calculator(predictions = predictions,On = 15 ,Off = 16 ,Power = 1900 )
df['Toaster'] =                       Predictions_calculator(predictions = predictions,On = 17 ,Off = 18 ,Power = 1000 )
df['Microwave'] =                     Predictions_calculator(predictions = predictions,On = 19 ,Off = 20 ,Power = 400 )
df['Computer_Monitor'] =              Predictions_calculator(predictions = predictions,On = 21 ,Off = 22 ,Power = 40 )
df['Audio_System'] =                  Predictions_calculator(predictions = predictions,On = 23 ,Off = 24 ,Power = 15 )
df['Vaccum_Cleaner'] =                Predictions_calculator(predictions = predictions,On = 25 ,Off = 26 ,Power = 1750 )
df['Hair_Dryer'] =                    Predictions_calculator(predictions = predictions,On = 27 ,Off = 28 ,Power = 350 )
df['Hair_Straighteners'] =            Predictions_calculator(predictions = predictions,On = 29 ,Off = 30 ,Power = 60 )
df['Clothing_Iron'] =                 Predictions_calculator(predictions = predictions,On = 31 ,Off = 32 ,Power = 1100 )
df['Oven'] =                          Predictions_calculator(predictions = predictions,On = 33 ,Off = 34 ,Power = 950 )
df['Fan'] =                           Predictions_calculator(predictions = predictions,On = 35 ,Off = 36 ,Power = 25 )
df['Washing_Machine'] =               Predictions_calculator(predictions = predictions,On = 37 ,Off = 38 ,Power = 450 )
df['EV'] =                            Predictions_calculator(predictions = predictions,On = 39 ,Off = 40 ,Power = 2000 )
df['Refrigerator'] =                  Predictions_calculator(predictions = predictions,On = 41 ,Off = 42 ,Power= 75 )
df['Electric_Water_Heater'] =         Predictions_calculator(predictions = predictions,On = 43 ,Off = 44 ,Power= 1125 )
df['Washing_Machine_2'] =             Predictions_calculator(predictions = predictions,On = 45 ,Off = 46 ,Power= 460 )


df['Total_Power'] = df.sum(axis=1)

df.to_csv(FILE_DIR_PATH/f'Results/{Load_Profile}/Predicted.csv', sep=';')

df = df.drop(columns=['Washing_Machine' ,'Wash_Dryer' ,'Dish_Washer' ,'EV' ,'Electric_Water_Heater' ,'Washing_Machine_2','Total_Power' ])

df.to_csv(FILE_DIR_PATH/f'Results/{Load_Profile}/Predicted_without_load_shifters.csv', sep=';')





start_date = f'2021-01-01'
#end_date = f'2021-12-31 23:59'
end_date = f'2021-01-01 23:59'
date_range = pd.date_range(start=start_date, end=end_date, freq='T')

df = pd.DataFrame(index=date_range)
df['predictions'] = predictions

def time_period_and_frequency ( predictions ,On ,Off ,Appliance_name ):

    Predicted_array = np.where((predictions == On) | (predictions == Off), predictions, 0)

    indices_On = np.where(Predicted_array == On)[0]
    indices_Off = np.where(Predicted_array == Off)[0]

    if len(indices_On) == 0 or len(indices_Off) == 0:
        return 0

    Time_period_list = []
    for i in range(len(indices_On)):
        Time_period = indices_Off[i] - indices_On[i]
        Time_period_list.append(Time_period)

    Average_Time_Period = np.mean(Time_period_list)

    return Average_Time_Period

Washing_Machine =   time_period_and_frequency( predictions = df['predictions'], On = 37, Off = 38, Appliance_name = 'Washing_Machine' )
Wash_Dryer =        time_period_and_frequency( predictions = df['predictions'], On = 3, Off = 4, Appliance_name = 'Wash_Dryer' )
Dish_Washer =       time_period_and_frequency( predictions = df['predictions'], On = 5, Off = 6, Appliance_name = 'Dish_Washer' )
EV =                time_period_and_frequency( predictions = df['predictions'], On = 39, Off = 40, Appliance_name = 'EV' )
Electric_Water =    time_period_and_frequency( predictions = df['predictions'], On = 43, Off = 44, Appliance_name = 'Electric_Water' )
Washing_Machine_2 = time_period_and_frequency( predictions = df['predictions'], On = 45, Off = 46, Appliance_name = 'Washing_Machine_2' )




# Group by month
df.index = pd.to_datetime(df.index)  # Ensure the index is a datetime type
monthly_groups = df.groupby(pd.Grouper(freq='M'))  # Group the data by month

# Store the results in a dictionary
results = {}

# Loop through each month
for month, data in monthly_groups:

    Washing_Machine = time_period_and_frequency(predictions=data['predictions'], On=37, Off=38,Appliance_name='Washing_Machine')
    Wash_Dryer = time_period_and_frequency(predictions=data['predictions'], On=3, Off=4, Appliance_name='Wash_Dryer')
    Dish_Washer = time_period_and_frequency(predictions=data['predictions'], On=5, Off=6, Appliance_name='Dish_Washer')
    EV = time_period_and_frequency(predictions=data['predictions'], On=39, Off=40, Appliance_name='EV')
    Electric_Water_Heater = time_period_and_frequency(predictions=data['predictions'], On=43, Off=44,Appliance_name='Electric_Water_Heater')
    Washing_Machine_2 = time_period_and_frequency(predictions=data['predictions'], On=45, Off=46,Appliance_name='Washing_Machine_2')

    # Store the results for the current month
    results[month.strftime('%Y-%m')] = {
        'Washing_Machine': Washing_Machine,
        'Wash_Dryer': Wash_Dryer,
        'Dish_Washer': Dish_Washer,
        'EV': EV,
        'Electric_Water_Heater': Electric_Water,
        'Washing_Machine_2': Washing_Machine_2
    }

    month_name = month.strftime('%Y-%m')

    df_for_DRL = pd.DataFrame()
    df_for_DRL['Appliance'] = ['Washing_Machine','Wash_Dryer','Dish_Washer','EV','Electric_Water_Heater','Washing_Machine_2']

    df_for_DRL['Run_Time'] = [results[f'{month_name}'][df_for_DRL['Appliance'][0]],
                              results[f'{month_name}'][df_for_DRL['Appliance'][1]],
                              results[f'{month_name}'][df_for_DRL['Appliance'][2]],
                              results[f'{month_name}'][df_for_DRL['Appliance'][3]],
                              results[f'{month_name}'][df_for_DRL['Appliance'][4]],
                              results[f'{month_name}'][df_for_DRL['Appliance'][5]]]

    df_for_DRL['Power'] = [450, 500, 1200, 2000, 1125, 460]

    # Filter out rows where 'Run_Time' is zero
    #df_for_DRL = df_for_DRL[df_for_DRL['Run_Time'] != 0]

    df_for_DRL = df_for_DRL.set_index('Appliance').T

    df_for_DRL.to_csv(FILE_DIR_PATH/f'Results/Database_Table_For_DRL/{Load_Profile}/{month_name}.csv',sep=';')









