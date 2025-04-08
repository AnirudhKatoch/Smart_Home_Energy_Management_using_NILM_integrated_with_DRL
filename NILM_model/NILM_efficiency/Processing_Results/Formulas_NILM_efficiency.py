from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

FILE_DIR_PATH = Path(__file__).parent.parent.parent.parent.parent


def self_consumption(csv_file_path):

    df = pd.read_feather(FILE_DIR_PATH / f'{csv_file_path}/df_absolute.feather')

    E_PVS = df['PV Power Plant (Generic) UI1.Pmeas_kW'].sum()/ (3600*1000)
    E_PVS2G = abs((df[df['Three-phase Meter GRID.POWER_P'] < 0]['Three-phase Meter GRID.POWER_P'].sum())/(3600*1000))
    Self_Consumption = ((E_PVS - E_PVS2G)/E_PVS)*100

    return Self_Consumption.round(4)

def degree_of_self_sufficiency(csv_file_path):

    df = pd.read_feather(FILE_DIR_PATH / f'{csv_file_path}/df_absolute.feather')
    E_G2L = abs((df[df['Three-phase Meter GRID.POWER_P'] > 0]['Three-phase Meter GRID.POWER_P'].sum())/(3600*1000))
    E_L = df['Variable Load (Generic) UI1.Pmeas_kW'].sum() / (3600*1000)
    degree_of_self_sufficiency = ((E_L - E_G2L) / E_L) * 100

    return degree_of_self_sufficiency.round(4)

def AC_System_Utilization_level(csv_file_path):

    df = pd.read_feather(FILE_DIR_PATH / f'{csv_file_path}/df_absolute.feather')

    Battery_discharge = df[df['Battery AC Power'] > 0]['Battery AC Power'].sum() / (3600 * 1000)
    Battery_charge = abs(df[df['Battery AC Power'] < 0]['Battery AC Power'].sum() / (3600 * 1000))

    Utilization_level = (Battery_discharge/Battery_charge)*100

    return Utilization_level.round(4)


def System_Utilization_level(csv_file_path):

    df = pd.read_feather(FILE_DIR_PATH / f'{csv_file_path}/df_absolute.feather')

    E_PV = df['PV Power Plant (Generic) UI1.Available_Ppv_kW'].sum()/ (3600*1000)
    E_PVS = df[df['PV Power Plant (Generic) UI1.Pmeas_kW'] > 0]['PV Power Plant (Generic) UI1.Pmeas_kW'].sum()/(3600*1000)
    E_AC2PVS = abs(df[df['PV Power Plant (Generic) UI1.Pmeas_kW'] < 0]['PV Power Plant (Generic) UI1.Pmeas_kW'].sum()/(3600*1000))

    E_BS2AC = df[df['Battery AC Power'] > 0]['Battery AC Power'].sum() / (3600 * 1000)
    E_AC2BS = abs(df[df['Battery AC Power'] < 0]['Battery AC Power'].sum() / (3600 * 1000))

    System_Utilization_level_value = ((E_PVS + E_BS2AC - E_AC2PVS - E_AC2BS )/E_PV) * 100

    return System_Utilization_level_value.round(4)

def Grid_feed_in(csv_file_path):

    df = pd.read_feather(FILE_DIR_PATH / f'{csv_file_path}/df_absolute.feather')
    Grid_Feed_in = (abs((df[df['Three-phase Meter GRID.POWER_P'] < 0]['Three-phase Meter GRID.POWER_P'].sum())/(3600*1000)))

    return Grid_Feed_in.round(4)

def Grid_supply(csv_file_path):

    df = pd.read_feather(FILE_DIR_PATH / f'{csv_file_path}/df_absolute.feather')
    Grid_Supply  = (abs((df[df['Three-phase Meter GRID.POWER_P'] > 0]['Three-phase Meter GRID.POWER_P'].sum())/(3600*1000)))

    return Grid_Supply.round(4)


def Balance_sheets_Costs(csv_file_path):

    df_profile = pd.read_feather(FILE_DIR_PATH / f'{csv_file_path}/df_absolute.feather')

    df = pd.read_csv(FILE_DIR_PATH / f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Typhoon_Inputs/Inputs_and_others_1_min_resolution.csv',sep=';')
    p_G2AC = (df['Electricity_Price_EUR_kWh'])     # EUR/kWh
    p_G2AC = (p_G2AC + 0.15)*1.19
    p_G2AC = p_G2AC.loc[p_G2AC.index.repeat(60)].reset_index(drop=True)

    p_AC2G = 0.08159 # EUR/kWh
    p_AC2G = [p_AC2G] * 86400

    Grid_Feed_in_RS = df_profile['Three-phase Meter GRID.POWER_P'].clip(upper=0) / (3600*1000)
    Grid_Supply_RS = df_profile['Three-phase Meter GRID.POWER_P'].clip(lower=0) / (3600*1000)

    Grid_feed_in_revenue_RS  = Grid_Feed_in_RS * p_AC2G
    Grid_Procurement_Costs_RS  = Grid_Supply_RS * p_G2AC

    Balance_Sheet_costs_RS  = Grid_Procurement_Costs_RS - Grid_feed_in_revenue_RS
    Balance_Sheet_costs_RS = Balance_Sheet_costs_RS.sum()

    return Balance_Sheet_costs_RS.round(4)


def Number_of_cycles(csv_path):

    df = pd.read_feather(FILE_DIR_PATH/f'{csv_path}/df_absolute.feather')
    Number_of_Cycles = abs(df['Battery DC Current']).sum()/(3600*109.042*2)

    return Number_of_Cycles

def Average_SOC_fucntion(csv_path):

    df = pd.read_feather(FILE_DIR_PATH/f'{csv_path}/df_absolute.feather')
    Average_SOC = (df['Battery SOC']).mean()

    return Average_SOC



csv_file_path_Actual = 'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/Actual/'
csv_file_path_10 = 'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/10/'
csv_file_path_20 = 'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/20/'
csv_file_path_30 = 'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/30/'
csv_file_path_40 = 'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/40/'
csv_file_path_50 = 'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/50/'
csv_file_path_60 = 'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/60/'
csv_file_path_70 = 'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/70/'
csv_file_path_80 = 'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/80/'
csv_file_path_90 = 'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/90/'
csv_file_path_100 = 'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/100/'







self_consumption = [self_consumption(csv_file_path_Actual),
                    self_consumption(csv_file_path_10),
                    self_consumption(csv_file_path_20),
                    self_consumption(csv_file_path_30),
                    self_consumption(csv_file_path_40),
                    self_consumption(csv_file_path_50),
                    self_consumption(csv_file_path_60),
                    self_consumption(csv_file_path_70),
                    self_consumption(csv_file_path_80),
                    self_consumption(csv_file_path_90),
                    self_consumption(csv_file_path_100),
                    ]


labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

fig1, ax1 = plt.subplots(figsize=(6.4,4.8))
ax1.bar(labels, self_consumption)
#ax1.set_title('Self Consumption Share')
ax1.set_xlabel('NILM model efficiency (%)',fontsize=15)
ax1.set_ylabel('Self Consumption Share (%)',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
plt.grid()
plt.tight_layout()
fig1.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Figures/self_consumption_share_NILM_accuracy.png')




Degree_of_self_sufficiency = [degree_of_self_sufficiency(csv_file_path_Actual),
                    degree_of_self_sufficiency(csv_file_path_10),
                    degree_of_self_sufficiency(csv_file_path_20),
                    degree_of_self_sufficiency(csv_file_path_30),
                    degree_of_self_sufficiency(csv_file_path_40),
                    degree_of_self_sufficiency(csv_file_path_50),
                    degree_of_self_sufficiency(csv_file_path_60),
                    degree_of_self_sufficiency(csv_file_path_70),
                    degree_of_self_sufficiency(csv_file_path_80),
                    degree_of_self_sufficiency(csv_file_path_90),
                    degree_of_self_sufficiency(csv_file_path_100),
                    ]

labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

fig1, ax1 = plt.subplots(figsize=(6.4,4.8))
ax1.bar(labels, Degree_of_self_sufficiency)
ax1.set_xlabel('NILM model efficiency (%)',fontsize=15)
ax1.set_ylabel('Degree of Self Sufficiency (%)',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
plt.grid()
plt.tight_layout()
fig1.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Figures/degree_of_self_sufficiency_NILM_accuracy.png')


AC_System_Utilization_level = [AC_System_Utilization_level(csv_file_path_Actual),
                    AC_System_Utilization_level(csv_file_path_10),
                    AC_System_Utilization_level(csv_file_path_20),
                    AC_System_Utilization_level(csv_file_path_30),
                    AC_System_Utilization_level(csv_file_path_40),
                    AC_System_Utilization_level(csv_file_path_50),
                    AC_System_Utilization_level(csv_file_path_60),
                    AC_System_Utilization_level(csv_file_path_70),
                    AC_System_Utilization_level(csv_file_path_80),
                    AC_System_Utilization_level(csv_file_path_90),
                    AC_System_Utilization_level(csv_file_path_100),
                    ]

labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

fig1, ax1 = plt.subplots(figsize=(6.4,4.8))
ax1.bar(labels, AC_System_Utilization_level)
ax1.set_xlabel('NILM model efficiency (%)',fontsize=15)
ax1.set_ylabel('AC System Utilization level (%)',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
plt.grid()
fig1.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Figures/AC_System_Utilization_level_NILM_accuracy.png')




System_Utilization_level = [System_Utilization_level(csv_file_path_Actual),
                    System_Utilization_level(csv_file_path_10),
                    System_Utilization_level(csv_file_path_20),
                    System_Utilization_level(csv_file_path_30),
                    System_Utilization_level(csv_file_path_40),
                    System_Utilization_level(csv_file_path_50),
                    System_Utilization_level(csv_file_path_60),
                    System_Utilization_level(csv_file_path_70),
                    System_Utilization_level(csv_file_path_80),
                    System_Utilization_level(csv_file_path_90),
                    System_Utilization_level(csv_file_path_100),
                    ]

labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

fig1, ax1 = plt.subplots(figsize=(6.4,4.8))
ax1.bar(labels, System_Utilization_level)
ax1.set_xlabel('NILM model efficiency (%)',fontsize=15)
ax1.set_ylabel('System Utilization level (%)',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
plt.grid()
fig1.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Figures/System_Utilization_level_NILM_accuracy.png')




Grid_feed_in = [Grid_feed_in(csv_file_path_Actual),
                    Grid_feed_in(csv_file_path_10),
                    Grid_feed_in(csv_file_path_20),
                    Grid_feed_in(csv_file_path_30),
                    Grid_feed_in(csv_file_path_40),
                    Grid_feed_in(csv_file_path_50),
                    Grid_feed_in(csv_file_path_60),
                    Grid_feed_in(csv_file_path_70),
                    Grid_feed_in(csv_file_path_80),
                    Grid_feed_in(csv_file_path_90),
                    Grid_feed_in(csv_file_path_100),
                    ]

labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

fig1, ax1 = plt.subplots(figsize=(6.4,4.8))
ax1.bar(labels, Grid_feed_in)
ax1.set_xlabel('NILM model efficiency (%)',fontsize=15)
ax1.set_ylabel('Grid feed in (kWh)',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
plt.grid()
fig1.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Figures/Grid_feed_in_NILM_accuracy.png')



Grid_supply = [Grid_supply(csv_file_path_Actual),
                    Grid_supply(csv_file_path_10),
                    Grid_supply(csv_file_path_20),
                    Grid_supply(csv_file_path_30),
                    Grid_supply(csv_file_path_40),
                    Grid_supply(csv_file_path_50),
                    Grid_supply(csv_file_path_60),
                    Grid_supply(csv_file_path_70),
                    Grid_supply(csv_file_path_80),
                    Grid_supply(csv_file_path_90),
                    Grid_supply(csv_file_path_100),
                    ]

labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

fig1, ax1 = plt.subplots(figsize=(6.4,4.8))
ax1.bar(labels, Grid_supply)
ax1.set_xlabel('NILM model efficiency (%)',fontsize=15)
ax1.set_ylabel('Grid supply (kWh)',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
plt.grid()
fig1.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Figures/Grid_supply_NILM_accuracy.png')



Balance_sheets_Costs = [Balance_sheets_Costs(csv_file_path_Actual),
                    Balance_sheets_Costs(csv_file_path_10),
                    Balance_sheets_Costs(csv_file_path_20),
                    Balance_sheets_Costs(csv_file_path_30),
                    Balance_sheets_Costs(csv_file_path_40),
                    Balance_sheets_Costs(csv_file_path_50),
                    Balance_sheets_Costs(csv_file_path_60),
                    Balance_sheets_Costs(csv_file_path_70),
                    Balance_sheets_Costs(csv_file_path_80),
                    Balance_sheets_Costs(csv_file_path_90),
                    Balance_sheets_Costs(csv_file_path_100),
                    ]

labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

fig1, ax1 = plt.subplots(figsize=(6.4,4.8))
ax1.bar(labels, Balance_sheets_Costs)
ax1.set_xlabel('NILM model efficiency (%)',fontsize=15)
ax1.set_ylabel('Balance sheets costs (Euro)',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
plt.grid()
fig1.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Figures/Balance_sheets_costs_NILM_accuracy.png')



Number_of_cycles = [Number_of_cycles(csv_file_path_Actual),
                    Number_of_cycles(csv_file_path_10),
                    Number_of_cycles(csv_file_path_20),
                    Number_of_cycles(csv_file_path_30),
                    Number_of_cycles(csv_file_path_40),
                    Number_of_cycles(csv_file_path_50),
                    Number_of_cycles(csv_file_path_60),
                    Number_of_cycles(csv_file_path_70),
                    Number_of_cycles(csv_file_path_80),
                    Number_of_cycles(csv_file_path_90),
                    Number_of_cycles(csv_file_path_100),
                    ]

labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

fig1, ax1 = plt.subplots(figsize=(6.4,4.8))
ax1.bar(labels, Number_of_cycles)
ax1.set_xlabel('NILM model efficiency (%)',fontsize=15)
ax1.set_ylabel('Battery Cycles',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
plt.grid()
fig1.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Figures/Number_of_cycles_NILM_accuracy.png')



