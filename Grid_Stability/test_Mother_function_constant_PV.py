from typhoon.api import hil
from typhoon.api.schematic_editor import model
import pytest
import logging
import typhoon.test.capture as cap
from pathlib import Path
import pandas as pd
import numpy as np

FILE_DIR_PATH = Path(__file__).parent.parent

#########################################################################################################################

# Actual

#########################################################################################################################

@pytest.fixture(scope="module")
def mother_function_inputs_a():
    mother_function_inputs_a = {

        'logger_a_path'      : 'Grid_Stability/Results/Average_day/0.2MVA/Actual/logger.log',
        'Load_path'          : 'Grid_Stability/Database/Average_inputs.csv',
        'PV_path'            : 'Grid_Stability/Database/Average_inputs.csv',
        'model_path'         : 'docs/Typhoon/Faster/Digital_twin_grid_stability_PV_constant_Load.tse',
        'Initial_battery_SOC': 0,
        'Control_Mode'       : 1,
        'Max_SOC'            : 100,
        'Min_SOC'            : 0,
        'capture_rate'       : 900,
        'Folder_name'        : 'Grid_Stability/Results/Average_day/0.2MVA/Actual/',
        'Testing'            : False,
        'Column_Load'        : 'Load_average_actual',
        'Column_PV'          : 'PV_average',
        'Loop_step_time'     : 0.9800208333333332,
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
    logger_a.info(f"model_path: {mother_function_inputs_a['model_path']}")
    logger_a.info(f"Initial_battery_SOC: {mother_function_inputs_a['Initial_battery_SOC']}")
    logger_a.info(f"Control_Mode: {mother_function_inputs_a['Control_Mode']}")
    logger_a.info(f"Max_SOC: {mother_function_inputs_a['Max_SOC']}")
    logger_a.info(f"Min_SOC: {mother_function_inputs_a['Min_SOC']}")
    logger_a.info(f"capture_rate: {mother_function_inputs_a['capture_rate']}")
    logger_a.info(f"Folder_name: {mother_function_inputs_a['Folder_name']}")
    logger_a.info(f"Column_Load: {mother_function_inputs_a['Column_Load']}")
    logger_a.info(f"Column_PV: {mother_function_inputs_a['Column_PV']}")
    logger_a.info(f"Loop_step_time: {mother_function_inputs_a['Loop_step_time']}")

    return logger_a

@pytest.fixture(scope="module")
def Load_data_a(mother_function_inputs_a, logger_a):

    Load_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_a['Load_path'], sep=';')
    Load_dataframe = np.array(Load_dataframe[mother_function_inputs_a['Column_Load']], dtype=np.float64)

    PV_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_a['PV_path'], sep=';')
    PV_dataframe = np.array(PV_dataframe[mother_function_inputs_a['Column_PV']], dtype=np.float64)

    return Load_dataframe, PV_dataframe

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

    hil.set_contactor('Grid Contactor', swControl=True, swState=True)
    hil.model_write('Grid UI1.Connect', 1)

    #   PV Power Plant

    hil.set_contactor('PV Contactor', swControl=True, swState=True)
    hil.model_write('PV Power Input', 0)

    #   Variable Load

    hil.set_contactor('Load Contactor', swControl=True, swState=True)
    hil.model_write('Load Power Input', 0)

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

@pytest.mark.parametrize('Test', ['Grid_Stability'])
def test_mother_a(setup_a, Test, mother_function_inputs_a, logger_a, Load_data_a):

    Load_dataframe_value, PV_dataframe_value = Load_data_a

    logger_a.info(f'Test : {Test}')
    test_time = len(Load_dataframe_value)
    logger_a.info(f'test_time : {test_time}')
    Total_time = round((test_time), 4)

    logger_a.info(f'It will take approximately {Total_time} seconds for the test {Test} to finish')

    hil.model_write('Load Power Input', Load_dataframe_value[0])
    hil.model_write('PV Power Input', PV_dataframe_value[0])

    hil.wait_sec(2)

    logger_a.info('Capture has started')

    cap.start_capture(duration=test_time,
                      rate=mother_function_inputs_a['capture_rate'],
                      signals=['Three-phase Meter Generation.POWER_P',

                               'Battery AC Power',
                               'Battery DC Power',
                               'Battery DC Voltage',
                               'Battery DC Current',
                               #'Battery Energy',
                               'Battery SOC',
                               'Residual Load',

                               'Load AC Power',

                               'Pmeas_kW_Grid',
                               'Vrms_meas_kV_Grid'
                               ], absolute_time=True)

    logger_a.info('Actual testing has started')

    for i, _ in enumerate(Load_dataframe_value):

        hil.model_write('Load Power Input', Load_dataframe_value[i])
        hil.model_write('PV Power Input', PV_dataframe_value[i])

        hil.wait_sec(mother_function_inputs_a['Loop_step_time'])

    logger_a.info('Loop Ended')

    df = cap.get_capture_results(wait_capture=True)
    df.reset_index(inplace=True)
    df.to_feather(FILE_DIR_PATH / f"{mother_function_inputs_a['Folder_name']}/df_{Test}.feather")
    # df.to_csv(FILE_DIR_PATH / f"{mother_function_inputs_a['Folder_name']}/df_{Test}.csv",sep=';')
    logger_a.info('Test has finished')



#########################################################################################################################

# Optimized Predicted

#########################################################################################################################

@pytest.fixture(scope="module")
def mother_function_inputs_b():
    mother_function_inputs_b = {

        'logger_b_path'      : 'Grid_Stability/Results/Average_day/0.2MVA/Optimized/logger.log',
        'Load_path'          : 'Grid_Stability/Database/Average_inputs.csv',
        'PV_path'            : 'Grid_Stability/Database/Average_inputs.csv',
        'model_path'         : 'docs/Typhoon/Faster/Digital_twin_grid_stability_PV_constant_Load.tse',
        'Initial_battery_SOC': 0,
        'Control_Mode'       : 1,
        'Max_SOC'            : 100,
        'Min_SOC'            : 0,
        'capture_rate'       : 900,
        'Folder_name'        : 'Grid_Stability/Results/Average_day/0.2MVA/Optimized/',
        'Testing'            : False,
        'Column_Load'        : 'Load_average_optimized',
        'Column_PV'          : 'PV_average',
        'Loop_step_time'     : 0.9800208333333332,
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
    logger_b.info(f"model_path: {mother_function_inputs_b['model_path']}")
    logger_b.info(f"Initial_battery_SOC: {mother_function_inputs_b['Initial_battery_SOC']}")
    logger_b.info(f"Control_Mode: {mother_function_inputs_b['Control_Mode']}")
    logger_b.info(f"Max_SOC: {mother_function_inputs_b['Max_SOC']}")
    logger_b.info(f"Min_SOC: {mother_function_inputs_b['Min_SOC']}")
    logger_b.info(f"capture_rate: {mother_function_inputs_b['capture_rate']}")
    logger_b.info(f"Folder_name: {mother_function_inputs_b['Folder_name']}")
    logger_b.info(f"Column_Load: {mother_function_inputs_b['Column_Load']}")
    logger_b.info(f"Column_PV: {mother_function_inputs_b['Column_PV']}")
    logger_b.info(f"Loop_step_time: {mother_function_inputs_b['Loop_step_time']}")

    return logger_b

@pytest.fixture(scope="module")
def Load_data_b(mother_function_inputs_b, logger_b):

    Load_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_b['Load_path'], sep=';')
    Load_dataframe = np.array(Load_dataframe[mother_function_inputs_b['Column_Load']], dtype=np.float64)

    PV_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_b['PV_path'], sep=';')
    PV_dataframe = np.array(PV_dataframe[mother_function_inputs_b['Column_PV']], dtype=np.float64)

    return Load_dataframe, PV_dataframe

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

    hil.set_contactor('Grid Contactor', swControl=True, swState=True)
    hil.model_write('Grid UI1.Connect', 1)

    #   PV Power Plant

    hil.set_contactor('PV Contactor', swControl=True, swState=True)
    hil.model_write('PV Power Input', 0)

    #   Variable Load

    hil.set_contactor('Load Contactor', swControl=True, swState=True)
    hil.model_write('Load Power Input', 0)

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

@pytest.mark.parametrize('Test', ['Grid_Stability'])
def test_mother_b(setup_b, Test, mother_function_inputs_b, logger_b, Load_data_b):

    Load_dataframe_value, PV_dataframe_value = Load_data_b

    logger_b.info(f'Test : {Test}')
    test_time = len(Load_dataframe_value)
    logger_b.info(f'test_time : {test_time}')
    Total_time = round((test_time), 4)
    logger_b.info(f'It will take approximately {Total_time} seconds for the test {Test} to finish')

    hil.model_write('Load Power Input', Load_dataframe_value[0] )
    hil.model_write('PV Power Input', PV_dataframe_value[0])

    hil.wait_sec(2)

    #if mother_function_inputs_b['Testing'] == True:
    #    Extra = 0
    #else:
    #    Extra = 100

    logger_b.info('Capture has started')

    cap.start_capture(duration=test_time,
                      rate=mother_function_inputs_b['capture_rate'],
                      signals=['Three-phase Meter Generation.POWER_P',

                               'Battery AC Power',
                               'Battery DC Power',
                               'Battery DC Voltage',
                               'Battery DC Current',
                               #'Battery Energy',
                               'Battery SOC',
                               'Residual Load',

                               'Load AC Power',

                               'Pmeas_kW_Grid',
                               'Vrms_meas_kV_Grid'
                               ], absolute_time=True)

    logger_b.info('Actual testing has started')

    for i, _ in enumerate(Load_dataframe_value):

        hil.model_write('Load Power Input', Load_dataframe_value[i])
        hil.model_write('PV Power Input', PV_dataframe_value[i])

        hil.wait_sec(mother_function_inputs_b['Loop_step_time'])

    logger_b.info('Loop Ended')

    df = cap.get_capture_results(wait_capture=True)
    df.reset_index(inplace=True)
    df.to_feather(FILE_DIR_PATH / f"{mother_function_inputs_b['Folder_name']}/df_{Test}.feather")
    # df.to_csv(FILE_DIR_PATH / f"{mother_function_inputs_b['Folder_name']}/df_{Test}.csv",sep=';')
    logger_b.info('Test has finished')

