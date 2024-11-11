import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FILE_DIR_PATH = Path(__file__).parent


df = pd.read_csv(FILE_DIR_PATH/f'Rewards/NILM_efficiency/Rewards_90.csv',sep=';')

plt.plot(df['episode_reward'])
plt.show()