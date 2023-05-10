"""Put plane on initial position and setup its flight condition
to our defined initial flight condition (altitude, attitude, velocity
, etc.)"""


import datetime
import msvcrt

from time import sleep
from math import sin, cos

import numpy as np
import pandas as pd
import tensorflow as tf

import xpc
from simutils import *
from normalization import DF_Nomalize, denorm, _norm

tf.compat.v1.disable_eager_execution()
print("Execute Eagerly: {}".format(tf.executing_eagerly()))


# V V -  -  -  -  -  -  -  -  -  -  V V
# V V          PARAMETERS           V V
# V V -  -  -  -  -  -  -  -  -  -  V V

INTERVAL = 0.05 # Seconds
DATA_PLOT_INTERVAL = 0.5 # Seconds
AIRPORT_LAT = 35.02684004546565 # Deg
AIRPORT_LON = -89.97248337952307 # Deg

GS_X = -89.97248337952307
GS_Y = 35.02684004546565

PAUSE_TOLERANCE = 10**(-5)

FEATURE_WINDOW_WIDTH = 1
# Features and labels for each model
ELV_FEATURE_COLUMNS = [
                    'alpha', 'gs', 'gs_d', 'gs_i', 'gamma_err', 'hdot',
                    #'alpha', 'gs', 'gs_d', 'gs_i', 'gamma_err',
                    'flap_0_bool', 'flap_1_bool', 'flap_2_bool', 'flap_3_bool', 'flap_4_bool', 'flap_5_bool', 'flap_6_bool']
ELV_SEQUENTIAL_LABELS = ["ctrl_col"]

AIL_FEATURE_COLUMNS = ['phi', 'phi_i', 'phi_d', 'loc', 'loc_i', 'loc_d']
AIL_SEQUENTIAL_LABELS = ["ctrl_whl"]

PLA_FEATURE_COLUMNS = [
                    'ias_err', 'gs', 'gs_i', #'gs_d',
                    'flap_0_bool', 'flap_1_bool', 'flap_2_bool', 'flap_3_bool', 'flap_4_bool', 'flap_5_bool', 'flap_6_bool']
PLA_SEQUENTIAL_LABELS = ["throttle"]

FLAP_FEATURE_COLUMNS = ['ias']
FLAP_SEQUENTIAL_LABELS = ["flap_rat"]

SAS_FEATURE_COLUMNS = ['hralt']
SAS_SEQUENTIAL_LABELS = ["sas"]

FEATURES = ELV_FEATURE_COLUMNS + PLA_FEATURE_COLUMNS + AIL_FEATURE_COLUMNS + FLAP_FEATURE_COLUMNS + SAS_FEATURE_COLUMNS

DATA_TO_RECORD = [
    'hbaro', 'hralt', 'theta', 'alpha', 'gs', 'loc', 'ias', 'tas', 'sas',
    'phi', 'elv_deg', 'ail1_deg', 'ail2_deg', 'rud1_deg', 'rud2_deg', 'n11',
    'hdot', 'onground', 'gs_d', 'gs_i', 'gamma_err', 'gs_stat', 'app_stat',
    'lat', 'lon', 'dist_m', 'gs_deg', 'gs_dev_deg', 'h_ref', 'h_err', 'loc_m',
    'ctrl_col', 'ctrl_whl', 'ctrl_rud', 'throttle', 'gear_bool', 'flap_rat'
]

DATA_TO_PLOT = [
    'dist_m', 'gs_deg', 'gs_dev_deg', 'h_ref', 'h_err', 'loc_m', 'sas', 'ias',
    'hralt', 'gs',
    'ctrl_col', 'ctrl_whl', 'ctrl_rud', 'throttle', 'flap_rat',
]


## Functions Definition
def Elv2LonStick(elv_def_deg: float) -> float:
    elv_def_deg = -0.6*elv_def_deg - 2.6
    if elv_def_deg >= 0:
        lon_stick = elv_def_deg / ELV_MAX
    else:
        lon_stick = elv_def_deg / ELV_MIN

    return lon_stick

def FlapProcessing(flap_to_send: float):
    flap_to_send = max (0.255 , min (1, round (flap_to_send * 6)/6))
    # if abs (flap_to_send - 0.333) < 0.01:
    #     flap_to_send = 0.255
    return flap_to_send

def get_data_from_xplane(
                client: xpc.XPlaneConnect,
                airport_lat: float = None,
                airport_lon: float = None
            ):
    # Getting data from x plane
    drefs_get = [
                'sim/flightmodel/position/alpha',                                   # Deg
                'sim/flightmodel/position/theta',                                   # Deg
                'sim/flightmodel/position/phi',                                     # Deg
                'sim/flightmodel/position/vpath',                                   # Deg
                'sim/flightmodel/position/true_airspeed',                           # mps
                'sim/flightmodel/position/indicated_airspeed',                      # kias
                'sim/cockpit/autopilot/airspeed',                                   # knots or mach, Selected airspeed
                'sim/flightmodel/position/vh_ind',                                  # mps
                'sim/weather/wind_speed_kts',                                       # Knots
                'sim/weather/wind_direction_degt',                                  # [0-359) degrees
            #    'sim/cockpit2/gauges/indicators/calibrated_airspeed_kts_pilot',    # Knots
                'sim/cockpit2/gauges/indicators/radio_altimeter_height_ft_pilot',   # Feet
            #    'sim/cockpit2/gauges/actuators/radio_altimeter_height_ft_pilot',   # Feet
            #    'sim/flightmodel/controls/elv1_def'                                # Deg
                'sim/flightmodel2/wing/elevator1_deg',
                'sim/flightmodel2/wing/aileron1_deg',
                'sim/flightmodel2/wing/aileron2_deg',
                #'sim/flightmodel2/wing/flap1_deg',
                #'sim/cockpit2/controls/flap_ratio',
                'sim/flightmodel/engine/ENGN_N1_',
                'sim/cockpit/radios/nav1_vdef_dot',
                'sim/cockpit/radios/nav1_hdef_dot',
                'sim/cockpit2/autopilot/glideslope_status',                         # 0=off,1=armed,2=captured
                'sim/cockpit2/autopilot/approach_status',                           # 0=off,1=armed,2=captured
                'sim/flightmodel/failures/onground_any',                            # 1 when aircraft on the ground
                'sim/flightmodel2/wing/rudder1_deg',
                'sim/flightmodel2/wing/rudder2_deg',
                'sim/flightmodel/position/psi',                                     # Deg
                'sim/flightmodel/position/true_psi',                                # Deg
                'sim/flightmodel/position/mag_psi',                                 # Deg
                'sim/flightmodel/position/hpath',                                   # Deg
                'sim/flightmodel/position/groundspeed',                             # mps
                'sim/flightmodel/position/P',                                       # Deg/sec
                'sim/flightmodel/position/Q',                                       # Deg/sec
                'sim/flightmodel/position/R',                                       # Deg/sec

            ]


    dref_values = client.getDREFs(drefs_get)

    posi = client.getPOSI()
    ctrl = client.getCTRL()


    # Process data
    rec_dict = {}

    rec_dict['lat'] = posi[0]       # Deg
    rec_dict['lon'] = posi[1]       # Deg
    rec_dict['dist_m'] = HaversineDistace(
                    posi[0], posi[1],
                    GS_Y, GS_X
                )


    if airport_lat and airport_lon:
        rec_dict['dist'] = HaversineDistace(
                    posi[0], airport_lat,
                    posi[1], airport_lon
                )

    rec_dict['alpha'] = dref_values[0][0] * deg2rad   # Rad
    rec_dict['theta'] = dref_values[1][0] * deg2rad   # Rad
    rec_dict['phi']   = dref_values[2][0] * deg2rad   # Rad
    rec_dict['vpath'] = dref_values[3][0] * deg2rad   # Rad

    rec_dict['tas']   = dref_values[4][0]               #*0.514444    # mps
    rec_dict['ias']   = dref_values[5][0]               # kias
    rec_dict['sas']   = dref_values[6][0]               # knots
    rec_dict['hdot']  = dref_values[7][0]               # mps
    rec_dict['hbaro'] = posi[2]                         # m
    rec_dict['hralt'] = dref_values[10][0] * 0.3048     # m

    rec_dict['elv_deg'] = dref_values[11][8]            # Deg
    rec_dict['ail1_deg']= dref_values[12][0]            # Deg
    rec_dict['ail2_deg']= dref_values[13][0]            # Deg

    rec_dict['rud1_deg']= dref_values[20][0]            # Deg
    rec_dict['rud2_deg']= dref_values[21][0]            # Deg

    rec_dict['psi']         = dref_values[22][0] * deg2rad  # Deg
    # rec_dict['mag_psi']     = dref_values[24][0] * deg2rad  # Deg
    rec_dict['hpath']       = dref_values[25][0] * deg2rad  # Deg
    rec_dict['groundspeed'] = dref_values[26][0]            # mps

    rec_dict['P'] = dref_values[27][0]            # Deg/sec
    rec_dict['Q'] = dref_values[28][0]            # Deg/sec
    rec_dict['R'] = dref_values[29][0]            # Deg/sec

    rec_dict['n11']     = dref_values[14][0]
    rec_dict['gs']      = dref_values[15][0]      # dot, 1 dot is 0.0875 DDM
    rec_dict['loc']     = dref_values[16][0]
    rec_dict['gs_stat'] = dref_values[17][0]
    rec_dict['app_stat']= dref_values[18][0]
    rec_dict['onground']= dref_values[19][0]

    rec_dict['ctrl_col'] = ctrl[0]
    rec_dict['ctrl_whl'] = ctrl[1]
    rec_dict['ctrl_rud'] = ctrl[2]
    rec_dict['throttle'] = ctrl[3]
    rec_dict['gear_bool']= ctrl[4]
    rec_dict['flap_rat'] = ctrl[5]

    rec_dict['gs_deg'] = atan(rec_dict['hralt']/rec_dict['dist_m']) * rad2deg
    rec_dict['gs_dev_deg'] = rec_dict['gs_deg'] - 3

    rec_dict['h_ref'] = rec_dict['dist_m'] * tan (3*deg2rad)
    rec_dict['h_err'] = rec_dict['hralt'] - rec_dict['h_ref']

    rec_dict['loc_m'] = DistanceToLine(rec_dict['lon'], rec_dict['lat'], m, c)
    if rec_dict['loc'] >= 0:
        rec_dict['loc_m'] = rec_dict['loc_m'] * -1

    return rec_dict


def predict_model(model, feat, feature_columns: list, label_colums: list, norm_param: dict, start_index: int = 0,
                window_width:int = FEATURE_WINDOW_WIDTH):

    feat_list = []
    for i in range(window_width):
        # Get features from recording
        _freq = int (1/INTERVAL)
        if start_index < 0 and (start_index+len(feature_columns)) == 0:
            _feat = feat[i-window_width][start_index:]
        else:
            _feat = feat[i-window_width][start_index:(start_index+len(feature_columns))] #
            #_feat = feat[-(i+1)][start_index:(start_index+len(feature_columns))] #
        # _feat = list(_feat)

        # Normalized Features
        for j, value in enumerate(_feat):
            _column = feature_columns[j]
            z, s = norm_param[_column]
            _feat[j] = _norm(value, z, s)

        feat_list.append(_feat)

    # Turn it into array and pass it to model
    feature = np.expand_dims (np.array(feat_list), axis=0)
    # label = model(feature).numpy()
    label = model.predict(feature)

    # Denormalized the labels
    label_send = [label[0, 0, i] for i in range (len(label_colums))]

    for i, _value in enumerate(label_send):
        z, s = norm_param[label_colums[i]]
        label_send[i] = denorm(_value, z, s)
    return label_send


def run(elv_model: tf.keras.Model, pla_model: tf.keras.Model,
        ail_model: tf.keras.Model, flap_model: tf.keras.Model,
        sas_model: tf.keras.Model,
        norm_param):

    lastUpdate = datetime.datetime.now()
    on_model = False
    is_running = True

    # Clear RealtimePlotter Directory
    garbages = os.listdir('RealtimePlotter')
    for garbage in garbages:
        os.remove(
            os.path.join('RealtimePlotter', garbage)
        )

    # Initial Setup
    time = 0
    last_height = 9999
    rec = []
    rec_plot = []
    feat= []

    print ("Begin establishing connection with X Plane")
    with xpc.XPlaneConnect(timeout=10000) as client:
        # Verify connection
        try:
            # If X-Plane does not respond to the request, a timeout error
            # will be raised.
            client.getDREF("sim/test/test_float")
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
            return

        feat_0 = [[0 for i in range(len(FEATURES))]]
        _ = predict_model(elv_model, feat_0, ELV_FEATURE_COLUMNS, ELV_SEQUENTIAL_LABELS,
                                 norm_param, 0)

        lastUpdate = datetime.datetime.now()
        sleep(INTERVAL)

        _ = input ("Press Enter to continue...")
        last_gs = 0
        gs_i = 0

        last_loc = 0
        loc_i = 0

        last_phi = 0
        phi_i = 0

        last_flap = 0

        sas = 200
        plot_time = 0
        gs_captured = False
        plot_file_exist = False
        last_plot_update = 0
        data_plot_index = 0

        # Control Loop
        while is_running == True:
            if msvcrt.kbhit():
                if str (msvcrt.getch()) == "b'q'":
                    print ("Quit recording")
                    is_running = False
                    break


            if (datetime.datetime.now() - lastUpdate).total_seconds() >= INTERVAL:
                # Retrieve data from x plane
                data_dict = get_data_from_xplane(client, AIRPORT_LAT, AIRPORT_LON)
                data_dict["sas"] = sas
                data_dict["ias_err"] = sas - data_dict["ias"]

                data_dict['gamma'] = data_dict['theta'] - data_dict['alpha']
                data_dict['gamma_err'] = -3*deg2rad - data_dict['gamma']

                data_dict["flap_0_bool"] = 1 if data_dict["flap_rat"] <  0.0833 else 0
                data_dict["flap_1_bool"] = 1 if data_dict["flap_rat"] >= 0.0833 and data_dict["flap_rat"] < 0.24  else 0
                data_dict["flap_2_bool"] = 1 if data_dict["flap_rat"] >= 0.24   and data_dict["flap_rat"] < 0.416 else 0
                data_dict["flap_3_bool"] = 1 if data_dict["flap_rat"] >= 0.416  and data_dict["flap_rat"] < 0.583 else 0
                data_dict["flap_4_bool"] = 1 if data_dict["flap_rat"] >= 0.583  and data_dict["flap_rat"] < 0.750 else 0
                data_dict["flap_5_bool"] = 1 if data_dict["flap_rat"] >= 0.750  and data_dict["flap_rat"] < 0.916 else 0
                data_dict["flap_6_bool"] = 1 if data_dict["flap_rat"] >= 0.916  else 0

                if abs (data_dict['gs']) < 1.25:
                    on_model = True
                    gs_captured = True

                data_dict['gs_captured'] = 0
                if gs_captured:
                    data_dict['gs_captured'] = 1

                # if abs(data_dict['gs']) < 2.45:
                #     on_model = True

                is_pause = True
                current_rec = []

                # Integral and Derivative Calculation
                if not gs_captured:
                    data_dict['gs_i'] = 0
                    data_dict['gs_d'] = 0
                    data_dict['loc_i'] = 0
                    data_dict['loc_d'] = 0
                    data_dict['phi_i'] = 0
                    data_dict['phi_d'] = 0
                else:
                    data_dict['gs_i'] = gs_i + INTERVAL * (last_gs + data_dict['gs']) / 2
                    data_dict['gs_d'] = (data_dict['gs'] - last_gs) #/ INTERVAL

                    data_dict['loc_i'] = loc_i + INTERVAL * (last_loc + data_dict['loc']) / 2
                    data_dict['loc_d'] = (data_dict['loc'] - last_loc) #/ INTERVAL

                    data_dict['phi_i'] = phi_i + INTERVAL * (last_phi + data_dict['phi']) / 2
                    data_dict['phi_d'] = (data_dict['phi'] - last_phi)  #/ INTERVAL



                for i, param in enumerate(DATA_TO_RECORD):
                    if time != 0:
                        if (abs (rec[-1][i+1] - data_dict[param]) > PAUSE_TOLERANCE
                            and not param.endswith('_d') and not param.endswith('_i')
                            ):
                            is_pause = False
                    else:
                        is_pause = False

                    current_rec.append(data_dict[param])

                # CONTINUE IF NOT PAUSING
                if is_pause:
                    print ("Paused")
                    lastUpdate = datetime.datetime.now()
                    continue

                gs_i = data_dict['gs_i']
                last_gs = data_dict['gs']
                loc_i = data_dict['loc_i']
                last_loc = data_dict['loc']
                phi_i = data_dict['phi_i']
                last_phi = data_dict['phi']


                # Record datas
                rec.append(
                    [time] + current_rec
                )

                # Record Feature
                current_feat = []
                for label in FEATURES:
                    current_feat.append(data_dict[label])
                feat.append(
                    current_feat
                )


                # MAKE PREDICTION AND SEND CONTROL TO X PLANE
                time += INTERVAL

                # if len (feat) < FEATURE_WINDOW_WIDTH:       # Skip if not ready yet
                if not on_model or len (feat) < FEATURE_WINDOW_WIDTH:
                    # Skip if not yet intercept glide slope
                    print ("Waiting Until Glide Slope Intercepted, gs: {}".format(data_dict['gs']))
                    lastUpdate = datetime.datetime.now()
                    continue

                # Selected Airspeed (sas)
                sas_send = predict_model(sas_model, feat, SAS_FEATURE_COLUMNS, SAS_SEQUENTIAL_LABELS,
                                        norm_param, -len(SAS_FEATURE_COLUMNS), window_width=1)
                sas_send = sas_send [0]

                # Flap
                flap_send = predict_model(flap_model,feat, FLAP_FEATURE_COLUMNS, FLAP_SEQUENTIAL_LABELS,
                                        norm_param, -len(SAS_FEATURE_COLUMNS)-len(FLAP_FEATURE_COLUMNS), window_width=1)
                flap_send = flap_send[0]

                # Ctrl. Column
                elv_send = predict_model(elv_model, feat, ELV_FEATURE_COLUMNS, ELV_SEQUENTIAL_LABELS,
                                        norm_param, 0)
                elv_send = elv_send[0]

                # Throttle
                thr_send = predict_model(pla_model, feat, PLA_FEATURE_COLUMNS, PLA_SEQUENTIAL_LABELS,
                                        norm_param, len (ELV_FEATURE_COLUMNS))
                thr_send = thr_send[0]

                # Ctrl. Wheel (Aileron)
                if len (feat) < 1:
                    ail_send = 0
                else:
                    ail_send = predict_model(ail_model, feat, AIL_FEATURE_COLUMNS, AIL_SEQUENTIAL_LABELS,
                                            norm_param, len (PLA_FEATURE_COLUMNS) + len (ELV_FEATURE_COLUMNS), window_width=1)
                    ail_send = ail_send[0]


                ail_send = min(1, max(-1, ail_send)) # 0.114
                thr_send = min(1, max(0 , thr_send))

                flap_send = max (last_flap, FlapProcessing(flap_send))
                gear_send = 1 if gs_captured else 0

                sas_send = max (140, min (160, sas_send))
                sas_send = 160 if (sas_send < 180 and sas_send > 150) else sas_send
                sas_send = 200 if (sas_send >=180) else sas_send
                sas_send = 140 if (sas_send <=150) else sas_send
                sas_send = 140 if (sas_send <=150) else sas_send
                sas_send = 140 if flap_send > 0.6  else sas_send
                sas = sas_send

                print ("gs: {:.4f}, gs_i: {:.4f}, gs_d: {:.4f}, loc: {:.4f}, elv: {:.4f}, thr: {:.4f}, ail: {:.4f}, flap: {:.3f}".format(data_dict['gs'], data_dict['gs_i'], data_dict['gs_d'], data_dict['loc'], elv_send, thr_send, ail_send, flap_send))

                # Send labels to x plane
                ctrl_send = [elv_send, ail_send, 0.00, thr_send, gear_send, flap_send]
                client.sendCTRL(ctrl_send)

                last_height = data_dict['hralt']
                last_flap = flap_send
                lastUpdate = datetime.datetime.now()

        # Save recordings into csv
        rec_np = np.array(rec)
        rec_df = pd.DataFrame(rec_np,
                    columns=(['time_s'] + DATA_TO_RECORD)
                    )
        rec_df.to_csv("rec.csv")


if __name__ == "__main__":

    # Select Model
    # print ("Select Elevator Model: ")
    # elv_model_path = SelectModelPrompt('Models')
    # print ("\nSelect Power Lever Model: ")
    # pla_model_path = SelectModelPrompt('Models')
    # print ("\nSelect Aileron Model: ")
    # ail_model_path = SelectModelPrompt('Models')
    # print ("\nSelect Flap Model: ")
    # flap_model_path = SelectModelPrompt('Models')
    # print ("\nSelect SAS Model: ")
    # sas_model_path = SelectModelPrompt('Models')

    elv_model_path = "Models\\Elevator" 
    pla_model_path = "Models\\Throttle"
    ail_model_path = "Models\\Aileron" 
    flap_model_path= "Models\\Flap"
    sas_model_path = "Models\\SelectetAirspeed"

    # Import model
    elv_model, _ = LoadModel(elv_model_path)
    pla_model, _ = LoadModel(pla_model_path)
    ail_model, _ = LoadModel(ail_model_path)
    flap_model, _= LoadModel(flap_model_path)
    sas_model, _ = LoadModel(sas_model_path)

    # Import train_DF to obtain normalization param.
    train_DF = pd.read_csv('Train_set.csv')
    train_DF, norm_param = DF_Nomalize(train_DF)

    ail_param = ['phi', 'phi_i', 'phi_d', 'loc', 'loc_i', 'loc_d', 'ctrl_whl']
    ail_train_DF = pd.read_csv('Train_set_ail.bak.csv')
    ail_train_DF, ail_norm_param = DF_Nomalize(ail_train_DF)

    for param in ail_param:
        norm_param[param] = ail_norm_param[param]

    print (norm_param)

    print ('min ctrl_rud: {}'.format(ail_train_DF['ctrl_rud'].min()))
    print ('max ctrl_rud: {}'.format(ail_train_DF['ctrl_rud'].max()))

    running = True
    while running:
        # Run simulation
        run(elv_model, pla_model, ail_model, flap_model, sas_model, norm_param)

        r = input ('Run again? (Y/n)')
        if str(r).lower() == "n":
            running = False
            break

    #
    print ('Simulation Finished')
