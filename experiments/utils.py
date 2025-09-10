import numpy as np


#Yaw only important for vehicles in 2d (rotation around the vertical axis)
def quaternion_to_yaw(q):
    
    w, x, y, z = q

    # yaw (z-axis rotation)
    yaw = np.arctan2(2.0 * (w*z + x*y),
                     1.0 - 2.0 * (y*y + z*z))
    return yaw
