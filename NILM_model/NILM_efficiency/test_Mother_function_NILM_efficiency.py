from typhoon.api import hil
from typhoon.api.schematic_editor import model
import pytest
import logging
import typhoon.test.capture as cap
from pathlib import Path
import pandas as pd
import numpy as np


FILE_DIR_PATH = Path(__file__).parent.parent.parent.parent


##############################################################################################################################

# NILM_efficiency

###############################################################################################################################

@pytest.fixture(scope="module")
def mother_function_inputs_a():

    NILM_efficiency = 'Actual'

    mother_function_inputs_a = {
        'logger_a_path':       f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/{NILM_efficiency}/logger.log',
        #'Load_path':           f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Databases/Total_Power_all_efficiencies_15_min.csv',
        'Load_path':           f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Databases/Load_Profile_actual_15_min.csv',
        'Irradiance_path':     f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Typhoon_Inputs/Inputs_and_others_15_min_resolution.csv',
        'model_path':          'docs/Typhoon/Faster/faster_Hybrid-inverter_VARTAPulseneo_Digital_twin_Real.tse',
        'Initial_battery_SOC': 50,
        'Control_Mode':        1,
        'Max_SOC':             100,
        'Min_SOC':             0,
        'capture_rate':        900,
        'Folder_name':         f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Results/{NILM_efficiency}',
        'Testing':             False,
        #'Column_Load':         f'Total_Power_{NILM_efficiency}',
        'Column_Load':         f'Total_Power',
        'Column_Irradiance':   'Irradiance',
        'Loop_step_time':      0.9803333333333333,
        'NILM_efficiency':     NILM_efficiency
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
    logger_a.info(f"Irradiance_path: {mother_function_inputs_a['Irradiance_path']}")
    logger_a.info(f"model_path: {mother_function_inputs_a['model_path']}")
    logger_a.info(f"Initial_battery_SOC: {mother_function_inputs_a['Initial_battery_SOC']}")
    logger_a.info(f"Control_Mode: {mother_function_inputs_a['Control_Mode']}")
    logger_a.info(f"Max_SOC: {mother_function_inputs_a['Max_SOC']}")
    logger_a.info(f"Min_SOC: {mother_function_inputs_a['Min_SOC']}")
    logger_a.info(f"capture_rate: {mother_function_inputs_a['capture_rate']}")
    logger_a.info(f"Folder_name: {mother_function_inputs_a['Folder_name']}")
    logger_a.info(f"Column_Load: {mother_function_inputs_a['Column_Load']}")
    logger_a.info(f"Column_Irradiance: {mother_function_inputs_a['Column_Irradiance']}")
    logger_a.info(f"Loop_step_time: {mother_function_inputs_a['Loop_step_time']}")
    logger_a.info(f"NILM_efficiency: {mother_function_inputs_a['NILM_efficiency']}")

    return logger_a


@pytest.fixture(scope="module")
def Load_data_a(mother_function_inputs_a,logger_a):

    Load_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_a['Load_path'],sep=';')
    Load_dataframe = np.array(Load_dataframe[mother_function_inputs_a['Column_Load']], dtype=np.float64)

    Irradiance_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_a['Irradiance_path'],sep=';')
    Irradiance_dataframe = np.array(Irradiance_dataframe[mother_function_inputs_a['Column_Irradiance']], dtype=np.float64)

    return Load_dataframe, Irradiance_dataframe


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

@pytest.mark.parametrize('Test', ['NILM_efficiency'])
def test_mother_a(setup_a, Test, mother_function_inputs_a, logger_a, Load_data_a):

    Load_dataframe_value, Irradiance_dataframe_value = Load_data_a

    P_nom = hil.read_analog_signal("Variable Load (Generic) UI1.Pnom_kW")
    Factor = 1 / (P_nom*1000)

    logger_a.info(f"Test : {mother_function_inputs_a['NILM_efficiency']}")
    test_time = len(Load_dataframe_value)
    logger_a.info(f'test_time : {test_time}')
    Total_time = round((test_time ), 4)
    logger_a.info(f'It will take approximately {Total_time} seconds for the test {Test} to finish')

    hil.model_write('Variable Load (Generic) UI1.Pref', Load_dataframe_value[0] * Factor)
    hil.model_write('PV Power Plant (Generic) UI1.Irradiance', Irradiance_dataframe_value[0])

    hil.wait_sec(2)

    if mother_function_inputs_a['Testing'] == True:
        Extra = 0
    else:
        Extra = 1

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
        hil.model_write('PV Power Plant (Generic) UI1.Irradiance', Irradiance_dataframe_value[i])

        hil.wait_sec(mother_function_inputs_a['Loop_step_time'])

    logger_a.info('Loop Ended')

    df = cap.get_capture_results(wait_capture=True)
    df.reset_index(inplace=True)
    df.to_feather(FILE_DIR_PATH / f"{mother_function_inputs_a['Folder_name']}/df_NILM_efficiency.feather")
    #df.to_csv(FILE_DIR_PATH / f"{mother_function_inputs_a['Folder_name']}/df_{Test}.csv",sep=';')
    logger_a.info('Test has finished')

