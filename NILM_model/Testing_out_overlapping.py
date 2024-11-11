import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FILE_DIR_PATH = Path(__file__).parent

df = pd.read_csv(FILE_DIR_PATH/f'Results/Optimized_predicted_load_profile.csv',sep=';')

Total_Power = df['Total_Power']

differences = Total_Power.diff()
#differences = differences.drop_duplicates()
differences.fillna(0, inplace=True)
#differences = differences.reset_index(drop=True)
# Filter out the specified values
differences = differences[~differences.isin([0,80,-80,500,-500,1200,-1200,-55,55,32,-32,100,-100,65,-65,1900,-1900,1000,-1000,400,-400,-40,40,15,-15,1750,-1750,350,-350,-60,60,1100,-1100,950,-950,25,-25,450,-450,-2000,2000,75,-75,1125,-1125,460,-460])]
differences = differences.reset_index(drop=True)
print(differences)