import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

FILE_DIR_PATH = Path(__file__).parent.parent

df_actual = pd.read_csv(FILE_DIR_PATH/f'Databases/Load_Profile_1.csv',sep=';')
#df_actual = df_actual[:60*24]

differences = df_actual['Total_Power'].diff()
differences.fillna(0, inplace=True)
differences = differences.values.reshape(-1, 1)

loaded_knn = joblib.load('knn_model.pkl')
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

df.to_csv(FILE_DIR_PATH/f'Results/Appliances_Predicted.csv',sep=';')


fig1,ax1 = plt.subplots(figsize=(20, 7))
ax1.plot(df.index/60, df['Boiler'],label = 'Boiler')
ax1.plot(df.index/60, df['Wash_Dryer'],label = 'Wash Dryer')
ax1.plot(df.index/60, df['Dish_Washer'],label = 'Dish Washer')
ax1.plot(df.index/60, df['Solar_Thermal_Pumping_Station'],label = 'Solar Thermal Pumping Station')
ax1.plot(df.index/60, df['Laptop_Computer'],label = 'Laptop Computer')
ax1.plot(df.index/60, df['Television'],label = 'Television')
ax1.plot(df.index/60, df['Home_Theater_PC'],label = 'Home Theater PC')
ax1.plot(df.index/60, df['Kettle'],label = 'Kettle')
ax1.plot(df.index/60, df['Toaster'],label = 'Toaster')
ax1.plot(df.index/60, df['Hair_Dryer'],label = 'Hair Dryer')
ax1.plot(df.index/60, df['Hair_Straighteners'],label = 'Hair Straighteners')
ax1.plot(df.index/60, df['Clothing_Iron'],label = 'Clothing Iron')
ax1.plot(df.index/60, df['Oven'],label = 'Oven')
ax1.plot(df.index/60, df['Fan'],label = 'Fan')
ax1.plot(df.index/60, df['Washing_Machine'],label = 'Washing Machine')
ax1.plot(df.index/60, df['EV'],label = 'EV')
ax1.plot(df.index/60, df['Refrigerator'],label = 'Refrigerator')
ax1.plot(df.index/60, df['Electric_Water_Heater'],label = 'Electric Water Heater')
ax1.plot(df.index/60, df['Washing_Machine_2'],label = 'Washing Machine 2')
ax1.set_xlim(min(df.index/60),max(df.index/60))
#plt.ylim(0,max(df['Total_Power'])+10)
#ax1.set_ylim(0,200)
ax1.set_xlabel('Time (Hours)')
ax1.set_ylabel('Power (W)')
#ax1.set_title('Load for a 24 hour period (Predicted)')
plt.grid()
plt.legend()
fig1.savefig(FILE_DIR_PATH/f'Figures/Appliance_predicted.png')
#plt.show()



fig2,ax2 = plt.subplots(figsize=(20, 7))
ax2.plot(df_actual.index/60, df_actual['Total_Power'],label = 'Total_Power')
ax2.set_xlim(min(df_actual.index/60),max(df_actual.index/60))
ax2.set_xlabel('Time (Hours)')
ax2.set_ylabel('Power (W)')
#ax2.set_title('Total power for a 24 hour period')
plt.grid()
plt.legend()
fig2.savefig(FILE_DIR_PATH/f'Figures/Total_power.png')
#plt.show()


fig3,ax3 = plt.subplots(figsize=(20, 7))
ax3.plot(df_actual.index/60, df_actual['Boiler'],label = 'Boiler')
ax3.plot(df_actual.index/60, df_actual['Wash_Dryer'],label = 'WashDryer')
ax3.plot(df_actual.index/60, df_actual['Dish_Washer'],label = 'DishWasher')
ax3.plot(df_actual.index/60, df_actual['Solar_Thermal_Pumping_Station'],label = 'Solar Thermal Pumping Station')
ax3.plot(df_actual.index/60, df_actual['Laptop_Computer'],label = 'Laptop Computer')
ax3.plot(df_actual.index/60, df_actual['Television'],label = 'Television')
ax3.plot(df_actual.index/60, df_actual['Home_Theater_PC'],label = 'Home Theater PC')
ax3.plot(df_actual.index/60, df_actual['Kettle'],label = 'Kettle')
ax3.plot(df_actual.index/60, df_actual['Toaster'],label = 'Toaster')
ax3.plot(df_actual.index/60, df_actual['Hair_Dryer'],label = 'Hair Dryer')
ax3.plot(df_actual.index/60, df_actual['Hair_Straighteners'],label = 'Hair Straighteners')
ax3.plot(df_actual.index/60, df_actual['Clothing_Iron'],label = 'Clothing Iron')
ax3.plot(df_actual.index/60, df_actual['Oven'],label = 'Oven')
ax3.plot(df_actual.index/60, df_actual['Fan'],label = 'Fan')
ax3.plot(df_actual.index/60, df_actual['Washing_Machine'],label = 'Washing Machine')
ax3.plot(df_actual.index/60, df_actual['EV'],label = 'EV')
ax3.plot(df_actual.index/60, df_actual['Refrigerator'],label = 'Refrigerator')
ax3.plot(df_actual.index/60, df_actual['Electric_Water_Heater'],label = 'Electric Water Heater')
ax3.plot(df_actual.index/60, df_actual['Washing_Machine_2'],label = 'Washing Machine 2')
ax3.set_xlim(min(df_actual.index/60),max(df.index/60))
#plt.ylim(0,max(df['Total_Power'])+10)
#ax1.set_ylim(0,200)
ax3.set_xlabel('Time (Hours)')
ax3.set_ylabel('Power (W)')
#ax3.set_title('Load for a 24 hour period (Actual)')
plt.grid()
plt.legend()
fig3.savefig(FILE_DIR_PATH/f'Figures/Appliance_actual.png')
#plt.show()





df_actual = df_actual.drop(columns=['Total_Power','Unnamed: 0'])

# Create an empty dictionary to store the MSE for each column
mse_dict = {}

# Loop over each column
for column in df.columns:
    # Compute the MSE for the current column
    mse = np.mean((df[column] - df_actual[column]) ** 2)

    #max_power = max(df_actual[column])

    # Store the result in the dictionary
    mse_dict[column] = mse



# Plot the MSE values as a histogram
plt.figure(figsize=(20, 7))

# Create a bar plot with the column names and their respective MSE values
plt.bar(mse_dict.keys(), mse_dict.values())

# Add labels and title
plt.xlabel('Appliance Names')
plt.ylabel('Mean Squared Error (W)/ Average Appliance Power (W)')
#plt.title('Predicted to Actual Appliance Power MSE')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.grid()

# Show the plot
plt.tight_layout()
plt.savefig(FILE_DIR_PATH/f'Figures/Histogram.png')
