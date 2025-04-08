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

        'logger_a_path'      : 'Grid_Stability/Voltages_Results/Average_day/0.2MVA/Actual/logger.log',
        'Load_path'          : 'Grid_Stability/Database/Average_day/5MVA/Residual_load.csv',
        'model_path'         : 'docs/Typhoon/Faster/Residual_load_typhoon.tse',
        'capture_rate'       : 900,
        'Folder_name'        : 'Grid_Stability/Voltages_Results/Average_day/0.2MVA/Actual/',
        'Testing'            : False,
        'Column_Load'        : 'Residual_Actual',
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
    logger_a.info(f"model_path: {mother_function_inputs_a['model_path']}")
    logger_a.info(f"capture_rate: {mother_function_inputs_a['capture_rate']}")
    logger_a.info(f"Folder_name: {mother_function_inputs_a['Folder_name']}")
    logger_a.info(f"Column_Load: {mother_function_inputs_a['Column_Load']}")
    logger_a.info(f"Loop_step_time: {mother_function_inputs_a['Loop_step_time']}")

    return logger_a

@pytest.fixture(scope="module")
def Load_data_a(mother_function_inputs_a, logger_a):

    Load_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_a['Load_path'], sep=';')
    Load_dataframe = np.array(Load_dataframe[mother_function_inputs_a['Column_Load']], dtype=np.float64)

    return Load_dataframe

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

    #   Residual Power

    hil.model_write('Residual Power Input', 0)

    hil.start_simulation()

    logger_a.info('Simulation of the model has started.')

    yield

    hil.stop_simulation()

@pytest.mark.parametrize('Test', ['Grid_Stability'])
def test_mother_a(setup_a, Test, mother_function_inputs_a, logger_a, Load_data_a):

    Load_dataframe_value = Load_data_a

    logger_a.info(f'Test : {Test}')
    test_time = len(Load_dataframe_value)
    logger_a.info(f'test_time : {test_time}')
    Total_time = round((test_time), 4)

    logger_a.info(f'It will take approximately {Total_time} seconds for the test {Test} to finish')

    hil.model_write('Residual Power Input', Load_dataframe_value[0])

    hil.wait_sec(2)

    logger_a.info('Capture has started')

    cap.start_capture(duration=test_time,
                      rate=mother_function_inputs_a['capture_rate'],
                      signals=['Residual AC Power',
                               'Pmeas_kW_Grid',
                               'Vrms_meas_kV_Grid'
                               ], absolute_time=True)

    logger_a.info('Actual testing has started')

    for i, _ in enumerate(Load_dataframe_value):

        hil.model_write('Residual Power Input', Load_dataframe_value[i])

        hil.wait_sec(mother_function_inputs_a['Loop_step_time'])

    logger_a.info('Loop Ended')

    df = cap.get_capture_results(wait_capture=True)
    df.reset_index(inplace=True)
    df.to_feather(FILE_DIR_PATH / f"{mother_function_inputs_a['Folder_name']}/df_{Test}.feather")
    # df.to_csv(FILE_DIR_PATH / f"{mother_function_inputs_a['Folder_name']}/df_{Test}.csv",sep=';')
    logger_a.info('Test has finished')



#########################################################################################################################

# Optimized

#########################################################################################################################

@pytest.fixture(scope="module")
def mother_function_inputs_b():

    mother_function_inputs_b = {

        'logger_b_path'      : 'Grid_Stability/Voltages_Results/Average_day/0.2MVA/Optimized/logger.log',
        'Load_path'          : 'Grid_Stability/Database/Average_day/5MVA/Residual_load.csv',
        'model_path'         : 'docs/Typhoon/Faster/Residual_load_typhoon.tse',
        'capture_rate'       : 900,
        'Folder_name'        : 'Grid_Stability/Voltages_Results/Average_day/0.2MVA/Optimized/',
        'Testing'            : False,
        'Column_Load'        : 'Residual_Optimized',
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
    logger_b.info(f"model_path: {mother_function_inputs_b['model_path']}")
    logger_b.info(f"capture_rate: {mother_function_inputs_b['capture_rate']}")
    logger_b.info(f"Folder_name: {mother_function_inputs_b['Folder_name']}")
    logger_b.info(f"Column_Load: {mother_function_inputs_b['Column_Load']}")
    logger_b.info(f"Loop_step_time: {mother_function_inputs_b['Loop_step_time']}")

    return logger_b

@pytest.fixture(scope="module")
def Load_data_b(mother_function_inputs_b, logger_b):

    Load_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_b['Load_path'], sep=';')
    Load_dataframe = np.array(Load_dataframe[mother_function_inputs_b['Column_Load']], dtype=np.float64)

    return Load_dataframe

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

    #   Residual Power

    hil.model_write('Residual Power Input', 0)

    hil.start_simulation()

    logger_b.info('Simulation of the model has started.')

    yield

    hil.stop_simulation()

@pytest.mark.parametrize('Test', ['Grid_Stability'])
def test_mother_b(setup_b, Test, mother_function_inputs_b, logger_b, Load_data_b):

    Load_dataframe_value = Load_data_b

    logger_b.info(f'Test : {Test}')
    test_time = len(Load_dataframe_value)
    logger_b.info(f'test_time : {test_time}')
    Total_time = round((test_time), 4)

    logger_b.info(f'It will take approximately {Total_time} seconds for the test {Test} to finish')

    hil.model_write('Residual Power Input', Load_dataframe_value[0])

    hil.wait_sec(2)

    logger_b.info('Capture has started')

    cap.start_capture(duration=test_time,
                      rate=mother_function_inputs_b['capture_rate'],
                      signals=['Residual AC Power',
                               'Pmeas_kW_Grid',
                               'Vrms_meas_kV_Grid'
                               ], absolute_time=True)

    logger_b.info('Actual testing has started')

    for i, _ in enumerate(Load_dataframe_value):

        hil.model_write('Residual Power Input', Load_dataframe_value[i])
        hil.wait_sec(mother_function_inputs_b['Loop_step_time'])

    logger_b.info('Loop Ended')

    df = cap.get_capture_results(wait_capture=True)
    df.reset_index(inplace=True)
    df.to_feather(FILE_DIR_PATH / f"{mother_function_inputs_b['Folder_name']}/df_{Test}.feather")
    # df.to_csv(FILE_DIR_PATH / f"{mother_function_inputs_b['Folder_name']}/df_{Test}.csv",sep=';')
    logger_b.info('Test has finished')