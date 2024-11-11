from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
import seaborn as sn
from sklearn.metrics import confusion_matrix

FILE_DIR_PATH = Path(__file__).parent.parent

############################################################################################################################

# Event Detector
def Event_maker(df,Appliance_name, On, Off):

    df_appliances = pd.read_csv(FILE_DIR_PATH/f'Databases/Load_Profile_k_NN_model_learn.csv',sep=';')

    Appliance = df_appliances[f'{Appliance_name}']
    differences = Appliance.diff()
    differences = differences.drop_duplicates()
    differences.fillna(0, inplace=True)
    differences = differences.reset_index(drop=True)

    Appliance_df = pd.DataFrame()
    Appliance_df['Event_detection'] = differences
    Appliance_df['Appliance_Detection'] = None

    mask =  (0 < Appliance_df['Event_detection'] )
    Appliance_df.loc[mask, 'Appliance_Detection'] = f'{Appliance_name} On'

    mask = (Appliance_df['Event_detection'] < 0 )
    Appliance_df.loc[mask, 'Appliance_Detection'] = f'{Appliance_name} Off'

    mask = (Appliance_df['Event_detection'] == 0 )
    Appliance_df.loc[mask, 'Appliance_Detection'] = f'Nothing'

    conditions = [
        (Appliance_df['Appliance_Detection'] == 'Nothing'),
        (Appliance_df['Appliance_Detection'] == f'{Appliance_name} On'),
        (Appliance_df['Appliance_Detection'] == f'{Appliance_name} Off')
    ]

    values = [0, On, Off]
    Appliance_df['Target'] = np.select(conditions, values, default=None)
    Appliance_df['Target'] = Appliance_df['Target'].astype('float64')

    combined_df = pd.concat([df, Appliance_df], ignore_index=True)

    return combined_df

df = pd.DataFrame()
df['Event_detection'] = None
df['Appliance_Detection'] = None
df['Target'] = None

df = Event_maker(df=df,Appliance_name='Boiler',On=1,Off=2)
df = Event_maker(df=df,Appliance_name='Wash_Dryer',On=3,Off=4)
df = Event_maker(df=df,Appliance_name='Dish_Washer',On=5,Off=6)
df = Event_maker(df=df,Appliance_name='Solar_Thermal_Pumping_Station',On=7,Off=8)
df = Event_maker(df=df,Appliance_name='Laptop_Computer',On=9,Off=10)
df = Event_maker(df=df,Appliance_name='Television',On=11,Off=12)
df = Event_maker(df=df,Appliance_name='Home_Theater_PC',On=13,Off=14)
df = Event_maker(df=df,Appliance_name='Kettle',On=15,Off=16)
df = Event_maker(df=df,Appliance_name='Toaster',On=17,Off=18)
df = Event_maker(df=df,Appliance_name='Microwave',On=19,Off=20)
df = Event_maker(df=df,Appliance_name='Computer_Monitor',On=21,Off=22)
df = Event_maker(df=df,Appliance_name='Audio_System',On=23,Off=24)
df = Event_maker(df=df,Appliance_name='Vaccum_Cleaner',On=25,Off=26)
df = Event_maker(df=df,Appliance_name='Hair_Dryer',On=27,Off=28)
df = Event_maker(df=df,Appliance_name='Hair_Straighteners',On=29,Off=30)
df = Event_maker(df=df,Appliance_name='Clothing_Iron',On=31,Off=32)
df = Event_maker(df=df,Appliance_name='Oven',On=33,Off=34)
df = Event_maker(df=df,Appliance_name='Fan',On=35,Off=36)
df = Event_maker(df=df,Appliance_name='Washing_Machine',On=37,Off=38)
df = Event_maker(df=df,Appliance_name='EV',On=39,Off=40)
df = Event_maker(df=df,Appliance_name='Refrigerator',On=41,Off=42)
df = Event_maker(df=df,Appliance_name='Electric_Water_Heater',On=43,Off=44)
df = Event_maker(df=df,Appliance_name='Washing_Machine_2',On=45,Off=46)

#df.to_csv(FILE_DIR_PATH/f'Results/Event_detection_result.csv',sep=';')


# Number of rows to repeat for each index
repeat_count = 1000

# Create a new DataFrame with the adjusted index
new_index = np.arange(len(df) * repeat_count)
new_data = []

for idx in range(len(df)):
    new_data.append(pd.DataFrame({
        'Event_detection': [df['Event_detection'][idx]] * repeat_count,
        'Appliance_Detection': [df['Appliance_Detection'][idx]] * repeat_count,
        'Target': [df['Target'][idx]] * repeat_count
    }))

# Concatenate all the new DataFrames into one
expanded_df = pd.concat(new_data, ignore_index=True)

#expanded_df.to_csv(FILE_DIR_PATH/f'Results/Event_detection_result_for_k_NN.csv',sep=';')

##########################################################################################################################

#k_NN Model

Appliance_df = expanded_df

X = Appliance_df['Event_detection']
y = Appliance_df['Target']
X = X.values.reshape(-1, 1)  # Reshape to (n_samples, 1)
y = y.values  # Convert to numpy array if not already
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

print(f'knn.score : {knn.score(X_test, y_test)}')

# Save the model
filename = 'knn_model.pkl'
#joblib.dump(knn, filename)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(20,20))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig(FILE_DIR_PATH/f'Figures/Confusion_matrix.png')


from sklearn.metrics import classification_report


# Assuming y_test and y_pred are available
# Generate the classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Convert the report to a DataFrame for easier visualization
classification_report_df = pd.DataFrame(report).transpose()



# Optionally, save the classification report to a CSV file
classification_report_df.to_csv('classification_report.csv',sep=';')