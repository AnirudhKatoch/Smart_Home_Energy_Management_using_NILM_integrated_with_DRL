from typhoon.api import hil
from typhoon.api.schematic_editor import model
import pytest
import logging
import typhoon.test.capture as cap
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pytz
from datetime import datetime
from deso_access.deso_controller import control_policy

FILE_DIR_PATH = Path(__file__).parent.parent.parent.parent


##############################################################################################################################

# DESO
# Load Profile 2 Actual

###############################################################################################################################

@pytest.fixture(scope="module")
def mother_function_inputs_a():

    mother_function_inputs_a = {
        'logger_a_path':       'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Actual/mit_DESO/Ideal/logger.log',
        'Load_path':           'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',
        'PV_path':             'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',
        'Time_stamp':          'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',
        'model_path':          'docs/Typhoon/Faster/faster_Hybrid-inverter_VARTAPulseneo_Digital_twin_Ideal.tse',
        'res_folder_path':     'NILM_and_DRL_based_EMS/DESO/res_files/Load_Profile_2/Actual/',
        'res_file_name':       'Day_',
        'Initial_battery_SOC': 50,
        'Control_Mode':        0,
        'Max_SOC':             100,
        'Min_SOC':             0,
        'capture_rate':        900,
        'Folder_name':         'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Actual/mit_DESO/Ideal/',
        'Testing':             False,
        'Column_Load':         'Load_actual_kW',
        'Column_PV':           'PV_Irradiance',
        'Column_Time_stamp':   'Time_unix',
        'Loop_step_time':      0.9554854166666669,
    }

    return mother_function_inputs_a


@pytest.fixture(scope="module")
def logger_a(mother_function_inputs_a):

    logger_a = logging.getLogger(__name__)
    logger_a.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(FILE_DIR_PATH / mother_function_inputs_a['logger_a_path'])
    file_handler.setFormatter(formatter)
    logger_a.addHandler(file_handler)

    logger_a.info(f"logger_a_path: {mother_function_inputs_a['logger_a_path']}")
    logger_a.info(f"Load_path: {mother_function_inputs_a['Load_path']}")
    logger_a.info(f"PV_path: {mother_function_inputs_a['PV_path']}")
    logger_a.info(f"Time_stamp: {mother_function_inputs_a['Time_stamp']}")
    logger_a.info(f"model_path: {mother_function_inputs_a['model_path']}")
    logger_a.info(f"res_folder_path: {mother_function_inputs_a['res_folder_path']}")
    logger_a.info(f"res_file_name: {mother_function_inputs_a['res_file_name']}")
    logger_a.info(f"Initial_battery_SOC: {mother_function_inputs_a['Initial_battery_SOC']}")
    logger_a.info(f"Control_Mode: {mother_function_inputs_a['Control_Mode']}")
    logger_a.info(f"Max_SOC: {mother_function_inputs_a['Max_SOC']}")
    logger_a.info(f"Min_SOC: {mother_function_inputs_a['Min_SOC']}")
    logger_a.info(f"capture_rate: {mother_function_inputs_a['capture_rate']}")
    logger_a.info(f"Folder_name: {mother_function_inputs_a['Folder_name']}")
    logger_a.info(f"Column_Load: {mother_function_inputs_a['Column_Load']}")
    logger_a.info(f"Column_PV: {mother_function_inputs_a['Column_PV']}")
    logger_a.info(f"Column_Time_stamp: {mother_function_inputs_a['Column_Time_stamp']}")
    logger_a.info(f"Loop_step_time: {mother_function_inputs_a['Loop_step_time']}")


    return logger_a

@pytest.fixture(scope="module")
def Load_data_a(mother_function_inputs_a,logger_a):

    Load_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_a['Load_path'],sep=';')
    Load_dataframe = np.array(Load_dataframe[mother_function_inputs_a['Column_Load']], dtype=np.float64)

    PV_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_a['PV_path'],sep=';')
    PV_dataframe = np.array(PV_dataframe[mother_function_inputs_a['Column_PV']], dtype=np.float64)

    Timestamp_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_a['Time_stamp'],sep=';')
    Timestamp_dataframe = np.array(Timestamp_dataframe[mother_function_inputs_a['Column_Time_stamp']], dtype=np.float64)
    Timestamp_dataframe = list(Timestamp_dataframe)

    res_dictionary_size = 365


    if mother_function_inputs_a['Testing'] == True:
        Load_dataframe = Load_dataframe[:96]
        PV_dataframe = PV_dataframe[:96]
        Timestamp_dataframe = Timestamp_dataframe[:96]
        res_dictionary_size = 5

    res_dictionary = {}

    for i in range(res_dictionary_size):
        file_path = FILE_DIR_PATH / mother_function_inputs_a['res_folder_path'] / f"{mother_function_inputs_a['res_file_name']}{i+1}.json"
        with open(file_path, 'r') as file:
            res_dictionary[f'res_{i}'] = json.load(file)

    logger_a.info(f'Length of res dictionary {len(res_dictionary)}')


    return Load_dataframe, PV_dataframe, Timestamp_dataframe, res_dictionary

@pytest.fixture(scope="module")
def setup_a(mother_function_inputs_a, logger_a):

    model_path = str(FILE_DIR_PATH / mother_function_inputs_a['model_path'])
    compiled_model_path = model.get_compiled_model_file(model_path)
    model.load(model_path)

    try:
        hw_settings = model.detect_hw_settings()
        vhil_device = False
        logger_a.info(f"{hw_settings[0]} {hw_settings[2]} device is used")
    except Exception:
        vhil_device = True
        logger_a.info("Virtual HIL device is used")

    model.compile()

    hil.load_model(compiled_model_path, vhil_device=vhil_device)

    #   Grid

    hil.model_write('Grid UI1.Connect', 1)
    hil.set_contactor('Grid Contactor', swControl=True, swState=True)

    #   PV Power Plant

    hil.model_write('PV Power Plant (Generic) UI1.Enable', 1)
    hil.set_contactor('PV Contactor', swControl=True, swState=True)
    hil.model_write('PV Power Plant (Generic) UI1.Irradiance', 0)

    #   Variable Load

    hil.set_scada_input_value('Variable Load (Generic) UI1.Enable', 1)
    hil.set_contactor('Load Contactor', swControl=True, swState=True)
    hil.model_write('Variable Load (Generic) UI1.Pref', 0)

    #   Battery ESS

    hil.set_contactor('Battery Contactor', swControl=True, swState=True)
    hil.model_write('Initial SOC', mother_function_inputs_a['Initial_battery_SOC'])
    hil.model_write('Max SOC', mother_function_inputs_a['Max_SOC'])
    hil.model_write('Min SOC', mother_function_inputs_a['Min_SOC'])
    hil.model_write('Control Mode', mother_function_inputs_a['Control_Mode'])
    hil.model_write('Battery Pref', 0)

    hil.start_simulation()

    logger_a.info('Simulation of the model has started.')

    yield

    hil.stop_simulation()

@pytest.mark.parametrize('Test', ['DESO'])
def test_mother_a(setup_a, Test, mother_function_inputs_a, logger_a, Load_data_a):

    Load_dataframe_value, PV_dataframe_value, Timestamp_dataframe_value, res_dictionary = Load_data_a

    P_nom = hil.read_analog_signal("Variable Load (Generic) UI1.Pnom_kW")
    Factor = 1 / (P_nom)

    logger_a.info(f'Test : {Test}')
    test_time = len(Load_dataframe_value)
    logger_a.info(f'test_time : {test_time}')
    Total_time = round((test_time ), 4)
    logger_a.info(f'It will take approximately {Total_time} seconds for the test {Test} to finish')

    hil.model_write('Variable Load (Generic) UI1.Pref', Load_dataframe_value[0] * Factor)
    hil.model_write('PV Power Plant (Generic) UI1.Irradiance', PV_dataframe_value[0])

    hil.wait_sec(2)

    if mother_function_inputs_a['Testing'] == True:
        Extra = 0
    else:
        Extra = 100

    logger_a.info('Capture has started')

    cap.start_capture(duration=test_time + Extra,
                      rate=mother_function_inputs_a['capture_rate'],
                      signals=['PV Power Plant (Generic) UI1.Available_Ppv_kW',
                               'PV Power Plant (Generic) UI1.Pmeas_kW',
                               'PV Power Plant (Generic) UI1.Vconv_rms_meas_V',

                               'Battery AC Power',
                               'Battery DC Power',
                               'Battery DC Voltage',
                               'Battery DC Current',
                               'Battery Energy',
                               'Battery SOC',
                               'Residual Load',

                               'Variable Load (Generic) UI1.Pmeas_kW',
                               'Variable Load (Generic) UI1.Vgrid_rms_meas_kV',
                               'Variable Load (Generic) UI1.Pref Probe',

                               'Three-phase Meter GRID.POWER_P',
                               'Three-phase Meter GRID.VAB_RMS',

                               ], absolute_time=True)

    logger_a.info('Actual testing has started')

    for i, _ in enumerate(Load_dataframe_value):

        hil.model_write('Variable Load (Generic) UI1.Pref', Load_dataframe_value[i] * Factor)
        hil.model_write('PV Power Plant (Generic) UI1.Irradiance', PV_dataframe_value[i])

        res = res_dictionary[f'res_{i // 96}']
        control = control_policy.PolicySchedule2D(res).get_control(datetime.fromtimestamp(Timestamp_dataframe_value[i],tz=pytz.utc),
                                                                   (hil.read_analog_signal('Battery SOC')/100),
                                                                   (hil.read_analog_signal('Residual Load')))
        hil.model_write('Battery Pref', control*1000)
        hil.wait_sec(mother_function_inputs_a['Loop_step_time'])

    logger_a.info('Loop Ended')

    df = cap.get_capture_results(wait_capture=True)
    df.reset_index(inplace=True)
    df.to_feather(FILE_DIR_PATH / f"{mother_function_inputs_a['Folder_name']}/df_{Test}.feather")
    #df.to_csv(FILE_DIR_PATH / f"{mother_function_inputs_a['Folder_name']}/df_{Test}.csv",sep=';')
    logger_a.info('Test has finished')



##############################################################################################################################

# DESO
# Load Profile 2 Optimized Predicted

###############################################################################################################################

@pytest.fixture(scope="module")
def mother_function_inputs_b():

    mother_function_inputs_b = {
        'logger_b_path':       'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Optimized_Predicted/mit_DESO/Ideal/logger.log',
        'Load_path':           'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',
        'PV_path':             'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',
        'Time_stamp':          'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',
        'model_path':          'docs/Typhoon/Faster/faster_Hybrid-inverter_VARTAPulseneo_Digital_twin_Ideal.tse',
        'res_folder_path':     'NILM_and_DRL_based_EMS/DESO/res_files/Load_Profile_2/Optimized_Predicted/',
        'res_file_name':       'Day_',
        'Initial_battery_SOC': 50,
        'Control_Mode':        0,
        'Max_SOC':             100,
        'Min_SOC':             0,
        'capture_rate':        900,
        'Folder_name':         'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Optimized_Predicted/mit_DESO/Ideal/',
        'Testing':             False,
        'Column_Load':         'Load_Optimized_predicted_kW',
        'Column_PV':           'PV_Irradiance',
        'Column_Time_stamp':   'Time_unix',
        'Loop_step_time':      0.9554854166666669,
    }

    return mother_function_inputs_b


@pytest.fixture(scope="module")
def logger_b(mother_function_inputs_b):

    logger_b = logging.getLogger(__name__)
    logger_b.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(FILE_DIR_PATH / mother_function_inputs_b['logger_b_path'])
    file_handler.setFormatter(formatter)
    logger_b.addHandler(file_handler)

    logger_b.info(f"logger_b_path: {mother_function_inputs_b['logger_b_path']}")
    logger_b.info(f"Load_path: {mother_function_inputs_b['Load_path']}")
    logger_b.info(f"PV_path: {mother_function_inputs_b['PV_path']}")
    logger_b.info(f"Time_stamp: {mother_function_inputs_b['Time_stamp']}")
    logger_b.info(f"model_path: {mother_function_inputs_b['model_path']}")
    logger_b.info(f"res_folder_path: {mother_function_inputs_b['res_folder_path']}")
    logger_b.info(f"res_file_name: {mother_function_inputs_b['res_file_name']}")
    logger_b.info(f"Initial_battery_SOC: {mother_function_inputs_b['Initial_battery_SOC']}")
    logger_b.info(f"Control_Mode: {mother_function_inputs_b['Control_Mode']}")
    logger_b.info(f"Max_SOC: {mother_function_inputs_b['Max_SOC']}")
    logger_b.info(f"Min_SOC: {mother_function_inputs_b['Min_SOC']}")
    logger_b.info(f"capture_rate: {mother_function_inputs_b['capture_rate']}")
    logger_b.info(f"Folder_name: {mother_function_inputs_b['Folder_name']}")
    logger_b.info(f"Column_Load: {mother_function_inputs_b['Column_Load']}")
    logger_b.info(f"Column_PV: {mother_function_inputs_b['Column_PV']}")
    logger_b.info(f"Column_Time_stamp: {mother_function_inputs_b['Column_Time_stamp']}")
    logger_b.info(f"Loop_step_time: {mother_function_inputs_b['Loop_step_time']}")

    return logger_b

@pytest.fixture(scope="module")
def Load_data_b(mother_function_inputs_b,logger_b):

    Load_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_b['Load_path'],sep=';')
    Load_dataframe = np.array(Load_dataframe[mother_function_inputs_b['Column_Load']], dtype=np.float64)

    PV_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_b['PV_path'],sep=';')
    PV_dataframe = np.array(PV_dataframe[mother_function_inputs_b['Column_PV']], dtype=np.float64)

    Timestamp_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_b['Time_stamp'],sep=';')
    Timestamp_dataframe = np.array(Timestamp_dataframe[mother_function_inputs_b['Column_Time_stamp']], dtype=np.float64)
    Timestamp_dataframe = list(Timestamp_dataframe)

    res_dictionary_size = 365

    if mother_function_inputs_b['Testing'] == True:
        Load_dataframe = Load_dataframe[:96]
        PV_dataframe = PV_dataframe[:96]
        Timestamp_dataframe = Timestamp_dataframe[:96]
        res_dictionary_size = 5

    res_dictionary = {}

    for i in range(res_dictionary_size):
        file_path = FILE_DIR_PATH / mother_function_inputs_b['res_folder_path'] / f"{mother_function_inputs_b['res_file_name']}{i+1}.json"
        with open(file_path, 'r') as file:
            res_dictionary[f'res_{i}'] = json.load(file)

    logger_b.info(f'Length of res dictionary {len(res_dictionary)}')


    return Load_dataframe, PV_dataframe, Timestamp_dataframe, res_dictionary

@pytest.fixture(scope="module")
def setup_b(mother_function_inputs_b, logger_b):

    model_path = str(FILE_DIR_PATH / mother_function_inputs_b['model_path'])
    compiled_model_path = model.get_compiled_model_file(model_path)
    model.load(model_path)

    try:
        hw_settings = model.detect_hw_settings()
        vhil_device = False
        logger_b.info(f"{hw_settings[0]} {hw_settings[2]} device is used")
    except Exception:
        vhil_device = True
        logger_b.info("Virtual HIL device is used")

    model.compile()

    hil.load_model(compiled_model_path, vhil_device=vhil_device)

    #   Grid

    hil.model_write('Grid UI1.Connect', 1)
    hil.set_contactor('Grid Contactor', swControl=True, swState=True)

    #   PV Power Plant

    hil.model_write('PV Power Plant (Generic) UI1.Enable', 1)
    hil.set_contactor('PV Contactor', swControl=True, swState=True)
    hil.model_write('PV Power Plant (Generic) UI1.Irradiance', 0)

    #   Variable Load

    hil.set_scada_input_value('Variable Load (Generic) UI1.Enable', 1)
    hil.set_contactor('Load Contactor', swControl=True, swState=True)
    hil.model_write('Variable Load (Generic) UI1.Pref', 0)

    #   Battery ESS

    hil.set_contactor('Battery Contactor', swControl=True, swState=True)
    hil.model_write('Initial SOC', mother_function_inputs_b['Initial_battery_SOC'])
    hil.model_write('Max SOC', mother_function_inputs_b['Max_SOC'])
    hil.model_write('Min SOC', mother_function_inputs_b['Min_SOC'])
    hil.model_write('Control Mode', mother_function_inputs_b['Control_Mode'])
    hil.model_write('Battery Pref', 0)

    hil.start_simulation()

    logger_b.info('Simulation of the model has started.')

    yield

    hil.stop_simulation()

@pytest.mark.parametrize('Test', ['DESO'])
def test_mother_b(setup_b, Test, mother_function_inputs_b, logger_b, Load_data_b):

    Load_dataframe_value, PV_dataframe_value, Timestamp_dataframe_value, res_dictionary = Load_data_b

    P_nom = hil.read_analog_signal("Variable Load (Generic) UI1.Pnom_kW")
    Factor = 1 / (P_nom)

    logger_b.info(f'Test : {Test}')
    test_time = len(Load_dataframe_value)
    logger_b.info(f'test_time : {test_time}')
    Total_time = round((test_time ), 4)
    logger_b.info(f'It will take approximately {Total_time} seconds for the test {Test} to finish')

    hil.model_write('Variable Load (Generic) UI1.Pref', Load_dataframe_value[0] * Factor)
    hil.model_write('PV Power Plant (Generic) UI1.Irradiance', PV_dataframe_value[0])

    hil.wait_sec(2)

    if mother_function_inputs_b['Testing'] == True:

        Extra = 0

    else:

        Extra = 100

    logger_b.info('Capture has started')

    cap.start_capture(duration=test_time + Extra,
                      rate=mother_function_inputs_b['capture_rate'],
                      signals=['PV Power Plant (Generic) UI1.Available_Ppv_kW',
                               'PV Power Plant (Generic) UI1.Pmeas_kW',
                               'PV Power Plant (Generic) UI1.Vconv_rms_meas_V',

                               'Battery AC Power',
                               'Battery DC Power',
                               'Battery DC Voltage',
                               'Battery DC Current',
                               'Battery Energy',
                               'Battery SOC',
                               'Residual Load',

                               'Variable Load (Generic) UI1.Pmeas_kW',
                               'Variable Load (Generic) UI1.Vgrid_rms_meas_kV',
                               'Variable Load (Generic) UI1.Pref Probe',

                               'Three-phase Meter GRID.POWER_P',
                               'Three-phase Meter GRID.VAB_RMS',

                               ], absolute_time=True)

    logger_b.info('Actual testing has started')

    for i, _ in enumerate(Load_dataframe_value):

        hil.model_write('Variable Load (Generic) UI1.Pref', Load_dataframe_value[i] * Factor)
        hil.model_write('PV Power Plant (Generic) UI1.Irradiance', PV_dataframe_value[i])

        res = res_dictionary[f'res_{i // 96}']
        control = control_policy.PolicySchedule2D(res).get_control(datetime.fromtimestamp(Timestamp_dataframe_value[i],tz=pytz.utc),
                                                                   (hil.read_analog_signal('Battery SOC')/100),
                                                                   (hil.read_analog_signal('Residual Load')))

        hil.model_write('Battery Pref', control*1000)

        hil.wait_sec(mother_function_inputs_b['Loop_step_time'])

    logger_b.info('Loop Ended')

    df = cap.get_capture_results(wait_capture=True)
    df.reset_index(inplace=True)
    df.to_feather(FILE_DIR_PATH / f"{mother_function_inputs_b['Folder_name']}/df_{Test}.feather")
    #df.to_csv(FILE_DIR_PATH / f"{mother_function_inputs_b['Folder_name']}/df_{Test}.csv",sep=';')
    logger_b.info('Test has finished')