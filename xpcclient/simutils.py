import os
import pickle

from math import sin, cos, atan2, sqrt, pi, atan, tan

import tensorflow as tf

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

def SelectModelPrompt(models_directory):

    models_list = ListModels(models_directory)
    models_list.sort()

    print ("Found {} models inside {}:".format(
                                        len(models_list), 
                                        models_directory
                                    ))
    
    print ("index    Model-name")
    for i, models_path in enumerate(models_list):
        number = "[{}]. ".format(i).ljust(7, " ")
        print ("  {}{}".format(
                            number, 
                            os.path.basename (models_path)
                        ))
    index = input("Please input your model's index (e.g 0): ")
    index = int(index)

    print ("You selected model {}".format(os.path.basename (models_list[index])))

    return models_list[index]

def LoadModel(path_to_model):
    """Load Model and optionally it's history as well"""
    history_file = os.path.join(path_to_model, 'history.pkl')
    model = tf.keras.models.load_model(path_to_model)
    # model = tf.saved_model.load(path_to_model)
    print ("model loaded")

    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    print ("model history loaded")

    return model, history