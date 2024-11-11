##############################################################################################################################

# Without_DESO 
# Load Profile 2 Actual Ideal

###############################################################################################################################

@pytest.fixture(scope="module")
def mother_function_inputs_c():
    mother_function_inputs_c = {

        'logger_c_path': 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Actual/Ideal/logger.log',
        'Load_path': 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',
        'PV_path': 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',
        'Time_stamp': 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',
        'model_path': 'docs/Typhoon/Faster/faster_Hybrid-faster_Hybrid-inverter_VARTAPulseneo_Digital_twin_Ideal.tse',
        'res_folder_path': 'NILM_and_DRL_based_EMS/DESO/res_files/Load_Profile_2/Actual/',
        'res_file_name': 'Day_',
        'Initial_battery_SOC': 50,
        'Control_Mode': 1,
        'Max_SOC': 100,
        'Min_SOC': 0,
        'capture_rate': 900,
        'Folder_name': 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Actual/Ideal/',
        'Testing': False,
        'Column_Load': 'Load_actual_kW',
        'Column_PV': 'PV_Irradiance',
        'Column_Time_stamp': 'Time_unix',
        'Loop_step_time': 0.9797708333333333,
    }

    return mother_function_inputs_c


@pytest.fixture(scope="module")
def logger_c(mother_function_inputs_c):
    logger_c = logging.getLogger(__name__)
    logger_c.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(FILE_DIR_PATH / mother_function_inputs_c['logger_c_path'])
    file_handler.setFormatter(formatter)
    logger_c.addHandler(file_handler)

    logger_c.info(f"logger_c_path: {mother_function_inputs_c['logger_c_path']}")
    logger_c.info(f"Load_path: {mother_function_inputs_c['Load_path']}")
    logger_c.info(f"PV_path: {mother_function_inputs_c['PV_path']}")
    logger_c.info(f"Time_stamp: {mother_function_inputs_c['Time_stamp']}")
    logger_c.info(f"model_path: {mother_function_inputs_c['model_path']}")
    logger_c.info(f"res_folder_path: {mother_function_inputs_c['res_folder_path']}")
    logger_c.info(f"res_file_name: {mother_function_inputs_c['res_file_name']}")
    logger_c.info(f"Initial_battery_SOC: {mother_function_inputs_c['Initial_battery_SOC']}")
    logger_c.info(f"Control_Mode: {mother_function_inputs_c['Control_Mode']}")
    logger_c.info(f"Max_SOC: {mother_function_inputs_c['Max_SOC']}")
    logger_c.info(f"Min_SOC: {mother_function_inputs_c['Min_SOC']}")
    logger_c.info(f"capture_rate: {mother_function_inputs_c['capture_rate']}")
    logger_c.info(f"Folder_name: {mother_function_inputs_c['Folder_name']}")
    logger_c.info(f"Column_Load: {mother_function_inputs_c['Column_Load']}")
    logger_c.info(f"Column_PV: {mother_function_inputs_c['Column_PV']}")
    logger_c.info(f"Column_Time_stamp: {mother_function_inputs_c['Column_Time_stamp']}")
    logger_c.info(f"Loop_step_time: {mother_function_inputs_c['Loop_step_time']}")

    return logger_c


@pytest.fixture(scope="module")
def Load_data_c(mother_function_inputs_c, logger_c):
    Load_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_c['Load_path'], sep=';')
    Load_dataframe = np.array(Load_dataframe[mother_function_inputs_c['Column_Load']], dtype=np.float64)

    PV_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_c['PV_path'], sep=';')
    PV_dataframe = np.array(PV_dataframe[mother_function_inputs_c['Column_PV']], dtype=np.float64)

    Timestamp_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_c['Time_stamp'], sep=';')
    Timestamp_dataframe = np.array(Timestamp_dataframe[mother_function_inputs_c['Column_Time_stamp']], dtype=np.float64)
    Timestamp_dataframe = list(Timestamp_dataframe)

    res_dictionary_size = 365

    if mother_function_inputs_c['Testing'] == True:
        Load_dataframe = Load_dataframe[:96]
        PV_dataframe = PV_dataframe[:96]
        Timestamp_dataframe = Timestamp_dataframe[:96]
        res_dictionary_size = 5

    res_dictionary = {}

    for i in range(res_dictionary_size):
        file_path = FILE_DIR_PATH / mother_function_inputs_c[
            'res_folder_path'] / f"{mother_function_inputs_c['res_file_name']}{i + 1}.json"
        with open(file_path, 'r') as file:
            res_dictionary[f'res_{i}'] = json.load(file)

    logger_c.info(f'Length of res dictionary {len(res_dictionary)}')

    return Load_dataframe, PV_dataframe, Timestamp_dataframe, res_dictionary


@pytest.fixture(scope="module")
def setup_c(mother_function_inputs_c, logger_c):
    model_path = str(FILE_DIR_PATH / mother_function_inputs_c['model_path'])
    compiled_model_path = model.get_compiled_model_file(model_path)
    model.load(model_path)

    try:
        hw_settings = model.detect_hw_settings()
        vhil_device = False
        logger_c.info(f"{hw_settings[0]} {hw_settings[2]} device is used")
    except Exception:
        vhil_device = True
        logger_c.info("Virtual HIL device is used")

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
    hil.model_write('Initial SOC', mother_function_inputs_c['Initial_battery_SOC'])
    hil.model_write('Max SOC', mother_function_inputs_c['Max_SOC'])
    hil.model_write('Min SOC', mother_function_inputs_c['Min_SOC'])
    hil.model_write('Control Mode', mother_function_inputs_c['Control_Mode'])
    hil.model_write('Battery Pref', 0)

    hil.start_simulation()

    logger_c.info('Simulation of the model has started.')

    yield

    hil.stop_simulation()


@pytest.mark.parametrize('Test', ['Without_DESO'])
def test_mother_c(setup_c, Test, mother_function_inputs_c, logger_c, Load_data_c):
    Load_dataframe_value, PV_dataframe_value, Timestamp_dataframe_value, res_dictionary = Load_data_c

    P_nom = hil.read_analog_signal("Variable Load (Generic) UI1.Pnom_kW")
    Factor = 1 / (P_nom)

    logger_c.info(f'Test : {Test}')
    test_time = len(Load_dataframe_value)
    logger_c.info(f'test_time : {test_time}')
    Total_time = round((test_time), 4)
    logger_c.info(f'It will take approximately {Total_time} seconds for the test {Test} to finish')

    hil.model_write('Variable Load (Generic) UI1.Pref', Load_dataframe_value[0] * Factor)
    hil.model_write('PV Power Plant (Generic) UI1.Irradiance', PV_dataframe_value[0])

    hil.wait_sec(2)

    if mother_function_inputs_c['Testing'] == True:
        Extra = 0
    else:
        Extra = 100

    logger_c.info('Capture has started')

    cap.start_capture(duration=test_time + Extra,
                      rate=mother_function_inputs_c['capture_rate'],
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

    logger_c.info('Actual testing has started')

    for i, _ in enumerate(Load_dataframe_value):
        hil.model_write('Variable Load (Generic) UI1.Pref', Load_dataframe_value[i] * Factor)
        hil.model_write('PV Power Plant (Generic) UI1.Irradiance', PV_dataframe_value[i])

        '''

        res = res_dictionary[f'res_{i // 96}']
        control = control_policy.PolicySchedule2D(res).get_control(
            datetime.fromtimestamp(Timestamp_dataframe_value[i], tz=pytz.utc),
            (hil.read_analog_signal('Battery SOC') / 100),
            (hil.read_analog_signal('Residual Load')))

        hil.model_write('Battery Pref', control * 1000)

        '''

        hil.wait_sec(mother_function_inputs_c['Loop_step_time'])

    logger_c.info('Loop Ended')

    df = cap.get_capture_results(wait_capture=True)
    df.reset_index(inplace=True)
    df.to_feather(FILE_DIR_PATH / f"{mother_function_inputs_c['Folder_name']}/df_{Test}.feather")
    # df.to_csv(FILE_DIR_PATH / f"{mother_function_inputs_c['Folder_name']}/df_{Test}.csv",sep=';')
    logger_c.info('Test has finished')


##############################################################################################################################

# Without DESO Real
# Load Profile 2 Optimized Predicted

###############################################################################################################################

@pytest.fixture(scope="module")
def mother_function_inputs_d():
    mother_function_inputs_d = {

        'logger_d_path': 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Optimized_Predicted/Ideal/logger.log',
        'Load_path': 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',
        'PV_path': 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',
        'Time_stamp': 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/Mother_function_inputs/Mother_function_inputs_data.csv',
        'model_path': 'docs/Typhoon/Faster/faster_Hybrid-faster_Hybrid-inverter_VARTAPulseneo_Digital_twin_Ideal.tse',
        'res_folder_path': 'NILM_and_DRL_based_EMS/DESO/res_files/Load_Profile_2/Optimized_Predicted/',
        'res_file_name': 'Day_',
        'Initial_battery_SOC': 50,
        'Control_Mode': 1,
        'Max_SOC': 100,
        'Min_SOC': 0,
        'capture_rate': 900,
        'Folder_name': 'NILM_and_DRL_based_EMS/DESO/SIL_Simulations/SIL_Results/Optimized_Predicted/Ideal/',
        'Testing': False,
        'Column_Load': 'Load_Optimized_predicted_kW',
        'Column_PV': 'PV_Irradiance',
        'Column_Time_stamp': 'Time_unix',
        'Loop_step_time': 0.9797708333333333,
    }

    return mother_function_inputs_d


@pytest.fixture(scope="module")
def logger_d(mother_function_inputs_d):
    logger_d = logging.getLogger(__name__)
    logger_d.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(FILE_DIR_PATH / mother_function_inputs_d['logger_d_path'])
    file_handler.setFormatter(formatter)
    logger_d.addHandler(file_handler)

    logger_d.info(f"logger_d_path: {mother_function_inputs_d['logger_d_path']}")
    logger_d.info(f"Load_path: {mother_function_inputs_d['Load_path']}")
    logger_d.info(f"PV_path: {mother_function_inputs_d['PV_path']}")
    logger_d.info(f"Time_stamp: {mother_function_inputs_d['Time_stamp']}")
    logger_d.info(f"model_path: {mother_function_inputs_d['model_path']}")
    logger_d.info(f"res_folder_path: {mother_function_inputs_d['res_folder_path']}")
    logger_d.info(f"res_file_name: {mother_function_inputs_d['res_file_name']}")
    logger_d.info(f"Initial_battery_SOC: {mother_function_inputs_d['Initial_battery_SOC']}")
    logger_d.info(f"Control_Mode: {mother_function_inputs_d['Control_Mode']}")
    logger_d.info(f"Max_SOC: {mother_function_inputs_d['Max_SOC']}")
    logger_d.info(f"Min_SOC: {mother_function_inputs_d['Min_SOC']}")
    logger_d.info(f"capture_rate: {mother_function_inputs_d['capture_rate']}")
    logger_d.info(f"Folder_name: {mother_function_inputs_d['Folder_name']}")
    logger_d.info(f"Column_Load: {mother_function_inputs_d['Column_Load']}")
    logger_d.info(f"Column_PV: {mother_function_inputs_d['Column_PV']}")
    logger_d.info(f"Column_Time_stamp: {mother_function_inputs_d['Column_Time_stamp']}")
    logger_d.info(f"Loop_step_time: {mother_function_inputs_d['Loop_step_time']}")

    return logger_d


@pytest.fixture(scope="module")
def Load_data_d(mother_function_inputs_d, logger_d):
    Load_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_d['Load_path'], sep=';')
    Load_dataframe = np.array(Load_dataframe[mother_function_inputs_d['Column_Load']], dtype=np.float64)

    PV_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_d['PV_path'], sep=';')
    PV_dataframe = np.array(PV_dataframe[mother_function_inputs_d['Column_PV']], dtype=np.float64)

    Timestamp_dataframe = pd.read_csv(FILE_DIR_PATH / mother_function_inputs_d['Time_stamp'], sep=';')
    Timestamp_dataframe = np.array(Timestamp_dataframe[mother_function_inputs_d['Column_Time_stamp']], dtype=np.float64)
    Timestamp_dataframe = list(Timestamp_dataframe)

    res_dictionary_size = 365

    if mother_function_inputs_d['Testing'] == True:
        Load_dataframe = Load_dataframe[:96]
        PV_dataframe = PV_dataframe[:96]
        Timestamp_dataframe = Timestamp_dataframe[:96]
        res_dictionary_size = 5

    res_dictionary = {}

    for i in range(res_dictionary_size):
        file_path = FILE_DIR_PATH / mother_function_inputs_d[
            'res_folder_path'] / f"{mother_function_inputs_d['res_file_name']}{i + 1}.json"
        with open(file_path, 'r') as file:
            res_dictionary[f'res_{i}'] = json.load(file)

    logger_d.info(f'Length of res dictionary {len(res_dictionary)}')

    return Load_dataframe, PV_dataframe, Timestamp_dataframe, res_dictionary


@pytest.fixture(scope="module")
def setup_d(mother_function_inputs_d, logger_d):
    model_path = str(FILE_DIR_PATH / mother_function_inputs_d['model_path'])
    compiled_model_path = model.get_compiled_model_file(model_path)
    model.load(model_path)

    try:
        hw_settings = model.detect_hw_settings()
        vhil_device = False
        logger_d.info(f"{hw_settings[0]} {hw_settings[2]} device is used")
    except Exception:
        vhil_device = True
        logger_d.info("Virtual HIL device is used")

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
    hil.model_write('Initial SOC', mother_function_inputs_d['Initial_battery_SOC'])
    hil.model_write('Max SOC', mother_function_inputs_d['Max_SOC'])
    hil.model_write('Min SOC', mother_function_inputs_d['Min_SOC'])
    hil.model_write('Control Mode', mother_function_inputs_d['Control_Mode'])
    hil.model_write('Battery Pref', 0)

    hil.start_simulation()

    logger_d.info('Simulation of the model has started.')

    yield

    hil.stop_simulation()


@pytest.mark.parametrize('Test', ['Without_DESO'])
def test_mother_d(setup_d, Test, mother_function_inputs_d, logger_d, Load_data_d):
    Load_dataframe_value, PV_dataframe_value, Timestamp_dataframe_value, res_dictionary = Load_data_d

    P_nom = hil.read_analog_signal("Variable Load (Generic) UI1.Pnom_kW")
    Factor = 1 / (P_nom)

    logger_d.info(f'Test : {Test}')
    test_time = len(Load_dataframe_value)
    logger_d.info(f'test_time : {test_time}')
    Total_time = round((test_time), 4)
    logger_d.info(f'It will take approximately {Total_time} seconds for the test {Test} to finish')

    hil.model_write('Variable Load (Generic) UI1.Pref', Load_dataframe_value[0] * Factor)
    hil.model_write('PV Power Plant (Generic) UI1.Irradiance', PV_dataframe_value[0])

    hil.wait_sec(2)

    if mother_function_inputs_d['Testing'] == True:

        Extra = 0

    else:

        Extra = 100

    logger_d.info('Capture has started')

    cap.start_capture(duration=test_time + Extra,
                      rate=mother_function_inputs_d['capture_rate'],
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

    logger_d.info('Actual testing has started')

    for i, _ in enumerate(Load_dataframe_value):
        hil.model_write('Variable Load (Generic) UI1.Pref', Load_dataframe_value[i] * Factor)
        hil.model_write('PV Power Plant (Generic) UI1.Irradiance', PV_dataframe_value[i])

        '''
        res = res_dictionary[f'res_{i // 96}']
        control = control_policy.PolicySchedule2D(res).get_control(
            datetime.fromtimestamp(Timestamp_dataframe_value[i], tz=pytz.utc),
            (hil.read_analog_signal('Battery SOC') / 100),
            (hil.read_analog_signal('Residual Load')))

        hil.model_write('Battery Pref', control * 1000)
        '''

        hil.wait_sec(mother_function_inputs_d['Loop_step_time'])

    logger_d.info('Loop Ended')

    df = cap.get_capture_results(wait_capture=True)
    df.reset_index(inplace=True)
    df.to_feather(FILE_DIR_PATH / f"{mother_function_inputs_d['Folder_name']}/df_{Test}.feather")
    # df.to_csv(FILE_DIR_PATH / f"{mother_function_inputs_d['Folder_name']}/df_{Test}.csv",sep=';')
    logger_d.info('Test has finished')
