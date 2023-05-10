"""Record Flight Data"""

import datetime
import msvcrt
import random

from time import sleep
from math import sin, cos

import numpy as np
import pandas as pd

import xpc
from simutils import *
from normalization import DF_Nomalize, denorm, _norm


## General Settings
INTERVAL = 0.05     # Seconds
THROTTLE_0  = 0.6

PAUSE_TOLERANCE = 10**(-5)
RECORDING_DIR   = "Records"

DATA_TO_RECORD = [
    'hbaro', 'hralt', 'theta', 'alpha', 'gs', 'loc', 'ias', 'tas', 'sas', 'groundspeed',
    'phi', 'psi', 'vpath', 'hpath', 'P', 'Q', 'R',
    'elv_deg', 'ail1_deg', 'ail2_deg', 'rud1_deg', 'rud2_deg', 'n11',
    'hdot', 'onground', 'gs_stat', 'app_stat',
    'lat', 'lon',
    'ctrl_col', 'ctrl_whl', 'ctrl_rud', 'throttle', 'gear_bool', 'flap_rat'
]

DATA_TO_PRINT = [
    'ail1_deg', 'ail2_deg', 'rud1_deg', 'rud2_deg', 'gs_stat', 'ias', 'gs', 'onground'
]

FLAP_SPEED = [
    220,            # FLAP UP
    210,            # FLAP 1
    180,            # FLAP 5
    170,            # FLAP 10
    160,            # FLAP 15
    150,            # FLAP 20
]


## Function Definitions
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

    return rec_dict


def autopilot(client: xpc.XPlaneConnect, rec_dict: dict):
    """Automatic flap deployment"""
    ias  = rec_dict['ias']
    flap = round (rec_dict['flap_rat'] * 6)
    gs_stat = round (rec_dict['gs_stat'])

    if flap < 6:
        to_send = [-998, -998]

        if ias <= FLAP_SPEED[flap]:
            to_send[1] = (flap + 1)/6

        if gs_stat == 2:
            to_send[0] = 1

        if not to_send == [-998, -998]:
            client.sendCTRL(
                [-998,  -998,  -998,  -998] + to_send
            )


def run(rec_num:int = 0):

    is_running = True

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

        time = 0
        lastUpdate = datetime.datetime.now()
        rec = []

        while is_running == True:
            # Listen to Quit Button
            if msvcrt.kbhit():
                if str (msvcrt.getch()) == "b'q'":
                    print ("Quit recording")
                    is_running = False
                    break

            # MAIN LOOP
            if (datetime.datetime.now() - lastUpdate).total_seconds() >= INTERVAL:

                rec_dict = get_data_from_xplane(client)
                autopilot(client, rec_dict)

                # Collect data to records
                # while also check if we are pausing
                is_pause = True
                current_rec = []
                for i, param in enumerate(DATA_TO_RECORD):
                    if time != 0:
                        if abs (rec[-1][i+1] - rec_dict[param]) > PAUSE_TOLERANCE:
                            is_pause = False
                    else:
                        is_pause = False
                    current_rec.append(rec_dict[param])

                # Only record if we are Pausing
                if not is_pause:
                    current_rec.insert(0, time)
                    rec.append(
                            current_rec
                        )
                    time += INTERVAL
                else:
                    print('paused.. ', end='')

                # Print Selected Parameters
                print ("{:.2f} \t".format(time), end='')
                for param in DATA_TO_PRINT:
                    print (" {}: {:.3f}".format(param, rec_dict[param]), end='')
                print()

                lastUpdate = datetime.datetime.now()

        # Save recordings into csv
        rec_np = np.array(rec)
        rec_df = pd.DataFrame(rec_np,
                    columns=(['time_s'] + DATA_TO_RECORD)
                    )
        rec_df.to_csv(
            os.path.join(
                RECORDING_DIR,
                "rec_{}.csv".format(rec_num)
            ), index=False)
        print ("Recording saved to {}".format(
                    os.path.join(
                        RECORDING_DIR,
                        "rec_{}.csv".format(rec_num)
                    )
                ))




if __name__ == "__main__":
    # Check if rec folder exist
    if not os.path.exists(RECORDING_DIR) or not os.path.isdir(RECORDING_DIR):
        os.makedirs(RECORDING_DIR)

    # Recording process
    while True:
        rec_num: int = int (input("Please input recording number: "))
        run (rec_num)

        res = input('\nDo you want to run again (Y/n)')
        if str (res).lower() == 'n':
            print("Quiting...")
            break
