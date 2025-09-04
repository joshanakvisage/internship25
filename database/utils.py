import numpy as np

def str_to_array(s):
    #Converts array in a string type (like movement,translation etc) to np.array for stonesoup
    if not s:
        return np.array([])
    return np.fromstring(s.strip("[]"), sep=',')