from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import CubicSpline

FILE_DIR_PATH = Path(__file__).parent.parent.parent.parent.parent

def own_consumption_share(csv_file_path):

    df = pd.read_feather(FILE_DIR_PATH / f'{csv_file_path}/df_absolute.feather')
    E_PVS = df['PV Power Plant (Generic) UI1.Pmeas_kW'].sum()/ (3600*1000)
    Grid_give = abs((df[df['Three-phase Meter GRID.POWER_P'] < 0]['Three-phase Meter GRID.POWER_P'].sum())/(3600*1000))
    own_consumption_share = ((E_PVS - Grid_give)/E_PVS)*100

    return own_consumption_share.round(4)


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


def Grid_feed_in_und_supply(csv_file_path):

    df = pd.read_feather(FILE_DIR_PATH / f'{csv_file_path}/df_absolute.feather')
    Grid_Feed_in_RS = (abs((df[df['Three-phase Meter GRID.POWER_P'] < 0]['Three-phase Meter GRID.POWER_P'].sum())/(3600*1000)))
    Grid_Supply_RS  = (abs((df[df['Three-phase Meter GRID.POWER_P'] > 0]['Three-phase Meter GRID.POWER_P'].sum())/(3600*1000)))

    return Grid_Feed_in_RS.round(4),Grid_Supply_RS.round(4)


def Balance_sheets_Costs(csv_file_path):

    df_profile = pd.read_feather(FILE_DIR_PATH / f'{csv_file_path}/df_absolute.feather')

    df = pd.read_csv(FILE_DIR_PATH / f'Feed in and electricity prices/Freiburg 2021/Electricity Price/energy-charts_Electricity_production_and_spot_prices_in_Germany_in_2021.csv',sep=';')
    p_G2AC = (df['Price (EUR/MWh, EUR/tCO2)']) / 1000    # EUR/kWh
    p_G2AC = (p_G2AC + 0.15) * 1.19
    p_G2AC = p_G2AC.loc[p_G2AC.index.repeat(3600)].reset_index(drop=True)

    p_AC2G = 0.08159 # EUR/kWh
    p_AC2G = [p_AC2G] * 31536000

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


def SPI_fucntion(csv_files_path_numerator,csv_files_path_denominator):

    csv_files_path_electricity_price = 'Feed in and electricity prices/Freiburg 2021/Electricity Price/'
    df = pd.read_csv(FILE_DIR_PATH / f'{csv_files_path_electricity_price}/energy-charts_Electricity_production_and_spot_prices_in_Germany_in_2021.csv',sep=';')


    p_G2AC = (df['Price (EUR/MWh, EUR/tCO2)']).mean() / 1000 # EUR/kWh
    p_G2AC = (p_G2AC + 0.15) * 1.19

    csv_file_path_feed_in_tarrif ='Feed in and electricity prices/Freiburg 2021/Feed in Tariff/'
    df = pd.read_csv(FILE_DIR_PATH / f'{csv_file_path_feed_in_tarrif}/Feed_in_Tariff_EUR_kWh.csv',sep=';')
    p_AC2G = df['Tariff_mean'].mean() # EUR/kWh


    # Real

    df = pd.read_feather(FILE_DIR_PATH / f'{csv_files_path_numerator}/df_absolute.feather')

    E_G2AC_Ref = df['Variable Load (Generic) UI1.Pmeas_kW'].sum() / (3600*1000)
    E_G2AC_Real = df[df['Three-phase Meter GRID.POWER_P'] > 0]['Three-phase Meter GRID.POWER_P'].sum()/(3600*1000)
    E_AC2G_Real = abs((df[df['Three-phase Meter GRID.POWER_P'] < 0]['Three-phase Meter GRID.POWER_P'].sum())/(3600*1000))

    Numerator = E_G2AC_Ref * p_G2AC - E_G2AC_Real * p_G2AC + E_AC2G_Real * p_AC2G

    print('E_G2AC_Ref',E_G2AC_Ref)
    print('E_G2AC_Real', E_G2AC_Real)
    print('E_AC2G_Real', E_AC2G_Real)

    print('Numerator',Numerator)

    # Ideal

    df = pd.read_feather(FILE_DIR_PATH / f'{csv_files_path_denominator}/df_absolute.feather')

    E_G2AC_Ref = df['Variable Load (Generic) UI1.Pmeas_kW'].sum() / (3600*1000)
    E_G2AC_Ideal = df[df['Three-phase Meter GRID.POWER_P'] > 0]['Three-phase Meter GRID.POWER_P'].sum()/(3600*1000)
    E_AC2G_Ideal = abs((df[df['Three-phase Meter GRID.POWER_P'] < 0]['Three-phase Meter GRID.POWER_P'].sum())/(3600*1000))

    Denominator = E_G2AC_Ref * p_G2AC - E_G2AC_Ideal * p_G2AC + E_AC2G_Ideal * p_AC2G

    SPI = ( Numerator / Denominator ) * 100

    return SPI.round(4)


csv_file_path_actual = 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Actual/Real/'
csv_file_path_optimized = 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Optimized_Predicted/Real/'


'''
own_consumption_share_actual = own_consumption_share(csv_file_path_actual)
own_consumption_share_optimized = own_consumption_share(csv_file_path_optimized)

fig1, ax1 = plt.subplots(figsize=(6.4,4.8))
ax1.bar(['Scenario 1','Scenario 2'], [own_consumption_share_actual,own_consumption_share_optimized], width = 0.50)
ax1.set_ylabel('Self Consumption Share (%)',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.grid(True)
ax1.set_ylim(0,100)
plt.tight_layout()
fig1.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Processing_Results/Figures/self_consumption_share_SIL_DESO_Appendix.png')


degree_of_self_sufficiency_actual = degree_of_self_sufficiency(csv_file_path_actual)
degree_of_self_sufficiency_optimized = degree_of_self_sufficiency(csv_file_path_optimized)

fig2, ax2 = plt.subplots(figsize=(6.4,4.8))
ax2.bar(['Scenario 1','Scenario 2'], [degree_of_self_sufficiency_actual,degree_of_self_sufficiency_optimized], width = 0.50)
ax2.set_ylabel('Degree of self sufficiency (%)', fontsize=15)
#ax2.set_title('Degree of self sufficiency ')
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
ax2.grid(True)
ax2.set_ylim(0,100)
plt.tight_layout()
fig2.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Processing_Results/Figures/degree_of_self_sufficiency_SIL_DESO_Appendix.png')


AC_System_Utilization_level_actual = AC_System_Utilization_level(csv_file_path_actual)
AC_System_Utilization_level_optimized = AC_System_Utilization_level(csv_file_path_optimized)

fig3, ax3 = plt.subplots(figsize=(6.4,4.8))
ax3.bar(['Scenario 1','Scenario 2'], [AC_System_Utilization_level_actual,AC_System_Utilization_level_optimized], width = 0.50)
ax3.set_ylabel('AC System Utilization level (%)',fontsize=15)
#ax3.set_title('AC System Utilization level')
ax3.tick_params(axis='x', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)
ax3.grid(True)
ax3.set_ylim(0,100)
plt.tight_layout()
fig3.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Processing_Results/Figures/AC_System_Utilization_level_SIL_DESO_Appendix.png')


System_Utilization_level_actual = System_Utilization_level(csv_file_path_actual)
System_Utilization_level_optimized = System_Utilization_level(csv_file_path_optimized)

fig4, ax4 = plt.subplots(figsize=(6.4,4.8))
ax4.bar(['Scenario 1','Scenario 2'], [System_Utilization_level_actual,System_Utilization_level_optimized], width = 0.50)
ax4.set_ylabel('System Utilization level (%)',fontsize=15)
#ax4.set_title('System Utilization level')
ax4.tick_params(axis='x', labelsize=15)
ax4.tick_params(axis='y', labelsize=15)
ax4.grid(True)
ax4.set_ylim(0,100)
plt.tight_layout()
fig4.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Processing_Results/Figures/System_Utilization_level_SIL_DESO_Appendix.png')

Grid_feed_in_actual, Grid_feed_supply_actual = Grid_feed_in_und_supply(csv_file_path_actual)
Grid_feed_in_actual_optimized, Grid_feed_supply_actual_optimized = Grid_feed_in_und_supply(csv_file_path_optimized)

fig5, ax5 = plt.subplots(figsize=(6.4,4.8))
label = ['Grid Feed in', 'Grid Supply']
Group_1 = [Grid_feed_in_actual, Grid_feed_in_actual_optimized]
Group_2 = [Grid_feed_supply_actual, Grid_feed_supply_actual_optimized]
num_groups = len(label)
bar_width = 0.275
index = np.arange(num_groups)
ax5.bar(index - bar_width / 2, [Group_1[0], Group_2[0]], bar_width,label = 'Scenario 1')
ax5.bar(index + bar_width / 2, [Group_1[1], Group_2[1]], bar_width,label = 'Scenario 2')
ax5.set_ylabel('Energy (kWh/a)',fontsize=15)
#ax5.set_title('Grid Feed in and procurement')
ax5.set_xticks(index)
ax5.set_xticklabels(label)
ax5.tick_params(axis='x', labelsize=15)
ax5.tick_params(axis='y', labelsize=15)
ax5.grid(True)
ax5.legend(fontsize=15)
plt.tight_layout()
fig5.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Processing_Results/Figures/Grid_supply_and_procurement_SIL_DESO_Appendix.png')


Balance_sheets_Costs_actual = Balance_sheets_Costs(csv_file_path_actual)
Balance_sheets_Costs_optimized = Balance_sheets_Costs(csv_file_path_optimized)

fig6, ax6 = plt.subplots(figsize=(6.4,4.8))
ax6.bar(['Scenario 1','Scenario 2'], [Balance_sheets_Costs_actual,Balance_sheets_Costs_optimized], width = 0.50)
ax6.set_ylabel(' Balance sheets costs (EUR/a)', fontsize=15)
#ax6.set_title('Profits')
ax6.tick_params(axis='x', labelsize=15)
ax6.tick_params(axis='y', labelsize=15)
#plt.subplots_adjust(top=0.98)
ax6.grid(True)
#ax6.set_ylim(0,100)
plt.tight_layout()
fig6.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Processing_Results/Figures/Balance_sheets_Costs_SIL_results_and_discussion.png')



Number_of_cycles_actual = Number_of_cycles(csv_file_path_actual)
Number_of_cycles_optimized = Number_of_cycles(csv_file_path_optimized)

fig7, ax7 = plt.subplots(figsize=(6.4,4.8))
ax7.bar(['Scenario 1','Scenario 2'], [Number_of_cycles_actual, Number_of_cycles_optimized], width = 0.50)
ax7.set_ylabel('Number of cycles', fontsize=15)
#ax7.set_title('Number of Battery Cycles')
ax7.tick_params(axis='x', labelsize=15)
ax7.tick_params(axis='y', labelsize=15)
ax7.grid(True)
#ax7.set_ylim(0,100)
plt.tight_layout()
fig7.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Processing_Results/Figures/Number_of_cycles_SIL_DESO_Appendix.png')



csv_files_path_numerator = 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Actual/mit_DESO/Real/'
csv_files_path_denominator = 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Actual/mit_DESO/Ideal/'

SPI_Actual = SPI_fucntion(csv_files_path_numerator,csv_files_path_denominator)

csv_files_path_numerator = 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Optimized_Predicted/mit_DESO/Real/'
csv_files_path_denominator = 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Optimized_Predicted/mit_DESO/Ideal/'

SPI_Optimized_Predicted = SPI_fucntion(csv_files_path_numerator,csv_files_path_denominator)

fig8, ax8 = plt.subplots(figsize=(6.4,4.8))
ax8.bar(['Scenario 1','Scenario 2'], [SPI_Actual, SPI_Optimized_Predicted], width = 0.50)
ax8.set_ylabel('System Performance Index (%)', fontsize=15)
#ax8.set_title('System Performance Index')
ax8.tick_params(axis='x', labelsize=15)
ax8.tick_params(axis='y', labelsize=15)
ax8.grid(True)
#ax8.set_ylim(98,102)
plt.tight_layout()
fig8.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Processing_Results/Figures/System_Performance_Index_SIL_DESO_Appendix.png')

'''
#degree_of_self_sufficiency_actual = degree_of_self_sufficiency(csv_file_path_actual)
#degree_of_self_sufficiency_optimized = degree_of_self_sufficiency(csv_file_path_optimized)
#Balance_sheets_Costs_actual = Balance_sheets_Costs(csv_file_path_actual)
#Balance_sheets_Costs_optimized = Balance_sheets_Costs(csv_file_path_optimized)

degree_of_self_sufficiency_actual = 70
degree_of_self_sufficiency_optimized = 80
Balance_sheets_Costs_actual = 1100
Balance_sheets_Costs_optimized = 900

# Data setup
scenarios = ['Scenario 1', 'Scenario 2']
x = np.arange(len(scenarios))  # label locations
width = 0.35  # width of the bars

fig, ax1 = plt.subplots(figsize=(6.4,4.8))

# Profits bar plot
ax1.bar(
    x - width / 2,
    [Balance_sheets_Costs_actual, Balance_sheets_Costs_optimized],
    width,
    label='Balance sheet costs',
    color='#ff7f0e',  # Light Salmon (soft orange)
    zorder=3  # Ensure bars are on top of grid
)
ax1.set_ylabel('Balance Sheet Costs (EUR/a)', fontsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.set_ylim(0, 1200)

# Degree of Self-Sufficiency bar plot
ax2 = ax1.twinx()
ax2.bar(
    x + width / 2,
    [degree_of_self_sufficiency_actual, degree_of_self_sufficiency_optimized],
    width,
    label='Degree of self sufficiency',
    color='#1f77b4',  # Light Sky Blue (soft blue)
    zorder=3  # Ensure bars are on top of grid
)
ax2.set_ylabel('Degree of Self Sufficiency (%)', fontsize=15)
ax2.set_ylim(0, 120)
ax2.tick_params(axis='y', labelsize=15)

# Set x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, fontsize=15)

# Legend and grid adjustments
fig.legend(fontsize=14, bbox_to_anchor=(0.87, 0.96725))
#ax1.grid(True, zorder=0)  # Set grid to lower zorder so bars appear on top
#plt.subplots_adjust(bottom= 0.06, top=0.98,right=0.875)
plt.tight_layout()

fig.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Processing_Results/Figures/Scenario 1 vs Scenario 2.png')


