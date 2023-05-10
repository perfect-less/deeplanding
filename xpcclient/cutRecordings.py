import math
import os
import pandas as pd
import argparse

from math import sin, cos, atan2, sqrt, pi, atan, tan

rad2deg = 180 / pi
deg2rad = pi / 180

feet2meter = 0.3048
EARTH_RADIUS = 6378145 # m

def HaversineDistace(lat1, lon1, lat2, lon2):
    """Calculate Distance between two point on earth in meters"""
    r = 6378145 # m
    
    lat1 = lat1 * deg2rad
    lat2 = lat2 * deg2rad
    lon1 = lon1 * deg2rad
    lon2 = lon2 * deg2rad

    h = sin( (lat2 - lat1)/2 )**2 + cos(lat1)*cos(lat2)*( sin( (lon2 - lon1)/2 )**2 )
    return 2*r*atan2( sqrt( h ), sqrt( 1-h ) )


def ClosestPointOnLine(x, y, m, c):
    x_p = (y + x/m - c)/(m+1/m)
    y_p = x_p*m + c

    return x_p, y_p

def LineIntersectionPoint(m1, c1, m2, c2):
    xi = (c2 - c1)/(m1 - m2)
    yi = xi*m1 + c1

    return xi, yi

def DistanceToLine(x, y, m, c):
    xp, yp = ClosestPointOnLine(x, y, m, c)

    dist = HaversineDistace(y, x, yp, xp)
    return dist

"""Basic Settings"""
INTERVAL = 0.05

RECORDING_DIR = "Records"
EXPORT_DIR = "Cleaned"

m = -46.780362224567945 
c = -4173.918522692542

GS_X = -89.97248337952307
GS_Y = 35.02684004546565

def cleanCSV(filename: str, written:bool=False, ailmode:bool=False):
    df = pd.read_csv(os.path.join(RECORDING_DIR, filename))

    # if df.loc[df.shape[0]-1, 'ctrl_rud'] != 0.00:
    #     print ("Skipped {}".format(filename))
    #     return

    begin_index = 0 if ailmode else df.loc[abs(df['gs']) < 1.25].index[0]
    end_index   = df.loc[df['hralt'] <= 30].index[0]

    new_df = pd.DataFrame (df.loc[begin_index:end_index, :])
    new_df.reset_index(inplace=True)
    new_df['gs_d'] = new_df["gs"].diff().fillna(0, inplace=False) #/ INTERVAL
    new_df['gs_i'] = new_df["gs"] * 0

    new_df['loc_d'] = new_df["loc"].diff().fillna(0, inplace=False) #/ INTERVAL
    new_df['loc_i'] = new_df["loc"] * 0

    new_df['phi_d'] = new_df["phi"].diff().fillna(0, inplace=False) #/ INTERVAL
    new_df['phi_i'] = new_df["phi"] * 0

    new_df['dist_m'] = new_df["phi"] * 0
    new_df['gs_deg'] = new_df["phi"] * 0
    new_df['gs_dev_deg'] = new_df["phi"] * 0
    new_df['h_ref'] = new_df["phi"] * 0
    new_df['h_err'] = new_df["phi"] * 0
    new_df['loc_m'] = new_df["phi"] * 0

    new_df['gamma'] = new_df['theta'] - new_df['alpha']
    new_df['gamma_err'] = -3*math.pi/180 - new_df['gamma']
    new_df['gamma_err_d'] = new_df["gamma_err"].diff().fillna(0, inplace=False) #/ INTERVAL

    new_df['ias_err'] = new_df['sas'] - new_df['ias']

    new_df["flap_0_bool"] = new_df["flap_rat"].apply(lambda x: 1 if x <  0.0833 else 0)
    new_df["flap_1_bool"] = new_df["flap_rat"].apply(lambda x: 1 if x >= 0.0833 and x < 0.24  else 0)
    new_df["flap_2_bool"] = new_df["flap_rat"].apply(lambda x: 1 if x >= 0.24   and x < 0.416 else 0)
    new_df["flap_3_bool"] = new_df["flap_rat"].apply(lambda x: 1 if x >= 0.416  and x < 0.583 else 0)
    new_df["flap_4_bool"] = new_df["flap_rat"].apply(lambda x: 1 if x >= 0.583  and x < 0.750 else 0)
    new_df["flap_5_bool"] = new_df["flap_rat"].apply(lambda x: 1 if x >= 0.750  and x < 0.916 else 0)
    new_df["flap_6_bool"] = new_df["flap_rat"].apply(lambda x: 1 if x >= 0.916  else 0)

    new_df['gs_captured'] = new_df["gs_stat"] - 1
    # captured_index = new_df.loc[round(new_df['gs_stat']) == 2].index[0]

    # for i in range (0, captured_index):
    #     new_df.loc[i, 'gs_d'] = 0
    #     new_df.loc[i, 'loc_d'] = 0
    # for i in range (captured_index+1, new_df.shape[0]):
    for i in range (0, new_df.shape[0]):
        if i != 0:
            new_df.loc[i, "gs_i"]  = new_df.loc[i-1, "gs_i"]  + 0.5*INTERVAL*(new_df.loc[i-1, "gs"]  + new_df.loc[i, "gs"])
            new_df.loc[i, "loc_i"] = new_df.loc[i-1, "loc_i"] + 0.5*INTERVAL*(new_df.loc[i-1, "loc"] + new_df.loc[i, "loc"])
            new_df.loc[i, "phi_i"] = new_df.loc[i-1, "phi_i"] + 0.5*INTERVAL*(new_df.loc[i-1, "phi"] + new_df.loc[i, "phi"])

        if False:
            lon = new_df.loc[i, 'lon']
            lat = new_df.loc[i, 'lat']

            xp, yp = ClosestPointOnLine(lon, lat, m, c)
            dist = HaversineDistace(yp, xp, GS_Y, GS_X)
            new_df.loc[i, 'dist_m'] = dist
            new_df.loc[i, 'gs_deg'] = atan(new_df.loc[i, 'hralt']/dist) * rad2deg
            new_df.loc[i, 'gs_dev_deg'] = new_df.loc[i, 'gs_deg'] - 3

            href = dist * tan (3*deg2rad)
            new_df.loc[i, 'h_ref'] = href
            new_df.loc[i, 'h_err'] = new_df.loc[i, 'hralt'] - href

            new_df.loc[i, 'loc_m'] = DistanceToLine(lon, lat, m, c)
            if new_df.loc[i, "loc"] > 0:
                new_df.loc[i, 'loc_m'] = -1*new_df.loc[i, 'loc_m']

    new_df.to_csv(os.path.join(EXPORT_DIR, filename), index=False)

    # if not written:
    #     new_df.to_csv(os.path.join(EXPORT_DIR, 'Train_all.csv'), index=False)
    # else:
    #     new_df.to_csv(os.path.join(EXPORT_DIR, 'Train_all.csv'), index=False, mode='a', header=False)


def create_parser(parser: argparse.ArgumentParser):
    prs = parser

    # Command flag
    prs.add_argument('-a', '--aileron', action="store_true",
                    help = "Trim aileron data")

    prs.add_argument('-i', '--in_dir', action="store",
                    help = """Directory of untrimemd flights data""")

    prs.add_argument('-o', '--out_dir', action="store",
                    help = """Directory to saved trimmed flights data""")

    return prs

if __name__ == "__main__":

    ailmode = False
    # Parsing Arguments
    parser = argparse.ArgumentParser()
    parser = create_parser(parser)

    args = parser.parse_args()

    if args.aileron:
        ailmode = True

    if not args.out_dir:
        print ("please specify out_dir")
        exit()

    EXPORT_DIR = str(args.out_dir)
    RECORDING_DIR = str(args.in_dir) if args.in_dir else RECORDING_DIR

    # Check if export folder exist
    if not os.path.exists(EXPORT_DIR) or not os.path.isdir(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)

    print ("Begin Processing...")
    written = False
    file_list = os.listdir(RECORDING_DIR)
    for file in file_list:
        if file.endswith('.csv'):
            cleanCSV(file, written, ailmode)
            if written == False:
                written = True
            print ("{} cleaned".format(file))
        else:
            continue

    print ("Finished")

