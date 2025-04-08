from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

FILE_DIR_PATH = Path(__file__).parent



csv_path = f'Results/Actual/'
df_Actual = pd.read_feather(FILE_DIR_PATH/f'{csv_path}/df_absolute.feather')
Residual_Actual = df_Actual['Residual Load']

csv_path = f'Results/Optimized_Predicted/'
df_Optimized = pd.read_feather(FILE_DIR_PATH/f'{csv_path}/df_absolute.feather')
Residual_Optimized = df_Optimized['Residual Load']

def ninetysix(series):
    averaged_data = series.groupby(series.index // 900).mean()
    return averaged_data

Residual_Actual = ninetysix(Residual_Actual) * 1000
Residual_Optimized = ninetysix(Residual_Optimized) * 1000

df = pd.DataFrame()

df['Residual_Actual'] = Residual_Actual
df['Residual_Optimized'] = Residual_Optimized

df.to_csv(FILE_DIR_PATH/f'Database/Residual_load.csv',sep=';')