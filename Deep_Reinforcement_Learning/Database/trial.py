import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FILE_DIR_PATH = Path(__file__).parent.parent.parent

PV_Power = pd.read_csv(FILE_DIR_PATH/'z/Standard_deviation_weg/csv_results/synPRO_el_family/Digital_Twin/mean/df_absolute.csv',sep=';')

plt.plot(PV_Power['PV Power Plant (Generic) UI1.Pmeas_kW'])
plt.show()


