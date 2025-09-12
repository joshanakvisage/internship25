
from stonesoup.models.measurement.linear import LinearGaussian
from enum import Enum
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity, ConstantAcceleration
import numpy as np
from stonesoup.types.state import GaussianState
from stonesoup.models.transition.nonlinear import ConstantTurn 
from datetime import timedelta, datetime


start_time = datetime(2025, 9, 12, 12, 0, 0) 

#Returns a translation model, a prior (compatable to the t mod.) and measurement model (compatable to the prior)
class Models(Enum):
    CONSTANT_VELOCITY = { #x_{k+1} = x_k + v_x T -> pedestrians, slow objects
        "trans_mod" : CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05 ),
                                                          ConstantVelocity(0.05 )]), 
        "prior" : GaussianState([[0], [0], [0], [0]], np.diag([0.5, 0.5, 0.5, 0.5]), timestamp=start_time), #krivo init treba povećat
        "meas_mod" : LinearGaussian(
            ndim_state=4,  # Number of state vectors (x_pos, x_vel, y_pos,y_vel)
            mapping=(0, 2),  # mapping so the measurement index fits the predicted state vector H_x*x_k-1 to form the measurement data form
            noise_covar=np.array([[0.1, 0], 
                                [0, 0.1]]) #where R is your measurement noise covariance.
            ) 
    }
    CONSTANT_ACCELERATION= { #PRIOR DEPENDS ON TRANSITION MODEL, MEASUREMENT MODEL DEPENDS ON PRIOR
        "trans_mod": CombinedLinearGaussianTransitionModel([
                                        ConstantAcceleration(0.05),  # Placeholder for acceleration model noise
                                        ConstantAcceleration(0.05)]),   # Placeholder for acceleration model noise  
                                        
        "prior": GaussianState([0, 0, 0, 0, 0, 0], np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), timestamp=start_time),
        "meas_mod": LinearGaussian(
                                        ndim_state=6,        
                                        mapping=(0, 3),      #TODO: CHECK IF POSITIONS ARE RIGHT measure x (0) and y (3) -> only position  
                                        noise_covar=np.array([[0.1, 0], #change to 3x3 if i insert velocity
                                                            [0, 0.1]])
                                    ),
    }
    
    #SINGER MODEL INSTED OF CONSTANT ACCELERATION -> UNNESECCARY
    
    COORDINATED_TURN = { #FOR CARS IN ROTORS, CARS IN LONG TURNS ETC
        "trans_mod": ConstantTurn(
            turn_noise_coeff=0.01,  # determines uncertainty in nonlinear turn rate -> Q process noise for turn rate ω
            linear_noise_coeffs=np.array([1, 1])   #determines uncertainty (process noise) in linear motion for velocity-> Q play with values
            # noise for [x, y, v, heading, ω]
        ),
        "prior":GaussianState(
                state_vector=np.array([[0],   # x position
                                    [1],   # x velocity
                                    [0],   # y position
                                    [1],   # y velocity
                                    [0]]), # turn rate ω
                covar=np.diag([0.5, 0.5, 0.5, 0.5, 0.01]),  # Confidence in initial guess
                timestamp=start_time),
        "meas_mod": LinearGaussian(
            ndim_state=5,             # number of states in ConstantTurn
            mapping=(0, 2),           # we can measure x and y positions from sensors
            noise_covar=np.array([[0.1, 0], 
                                [0, 0.1]])  # measurement noise for x and y -> R 
        )
    }

    #NOT NEEDED FOR NOW
    CONSTANT_TURN_ACCELERATION = {
        "trans_mod": 0,
        "prior":0,
        "meas_mod":0
    }