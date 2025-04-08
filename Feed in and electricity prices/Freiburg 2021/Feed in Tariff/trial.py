import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

FILE_DIR_PATH = Path(__file__).parent


df = pd.read_csv(FILE_DIR_PATH/'Feed_in_Tariff_EUR_kWh_old_wrong.csv',sep=';')
df = df.drop(columns=['Unnamed: 0'])

columns_to_update = ['Tariff_mean', 'Tariff_max_1', 'Tariff_min_1',
                     'Tariff_max_2', 'Tariff_min_2', 'Tariff_max_3', 'Tariff_min_3']


df[columns_to_update] = 0.0816
df.to_csv(FILE_DIR_PATH / f'Feed_in_Tariff_EUR_kWh.csv',sep=';')
print(df)
