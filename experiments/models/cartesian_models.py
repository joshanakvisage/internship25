
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
        "trans_mod" : CombinedLinearGaussianTransitionModel([ConstantVelocity(0.001),
                                                          ConstantVelocity(0.001)]),  # Q
        "prior" : GaussianState([[0], [0], [0], [0]], np.diag([0.5, 0.5, 0.5, 0.5]), timestamp=start_time), #covar = P 
        "meas_mod" : LinearGaussian(
            ndim_state=4,  # Number of state vectors (x_pos, x_vel, y_pos,y_vel)
            mapping=(0, 2),  # mapping so the measurement index fits the predicted state vector H_x*x_k-1 to form the measurement data form
            noise_covar=np.array([[11, 0], 
                                [0, 11]]) #where R is your measurement noise covariance.
            ) 
    }
    
    COORDINATED_TURN = { #FOR CARS IN ROTORS, CARS IN LONG TURNS ETC
        "trans_mod": ConstantTurn(
            turn_noise_coeff=0.1,  # determines uncertainty in nonlinear turn rate -> Q process noise for turn rate ω Q 
            linear_noise_coeffs=np.array([0.1, 0.1])   #determines uncertainty (process noise) in linear motion for velocity-> Q play with values
            # noise for [x, y, v, heading, ω] 
        ),
        "prior":GaussianState(
                state_vector=np.array([[0],   # x position
                                    [10],   # x velocity
                                    [0],   # y position
                                    [15],   # y velocity
                                    [0.1]]), # turn rate ω
                covar=np.diag([1., 10., 1., 10., 10.]),  # Confidence in initial guess, is updated = P uncertainty
                timestamp=start_time),
        "meas_mod": LinearGaussian(
            ndim_state=5,             # number of states in ConstantTurn
            mapping=(0, 2),           # we can measure x and y positions from sensors
            noise_covar=np.array([[1, 0], 
                                [0, 1]])  # measurement noise for x and y -> R 
        )
    }


def generate_cartesian_prior_from_mes(z0: np.ndarray,  vx0: float = -8., vy0: float = -10., w0: float = -0.1):
    """
    Generates a GaussianState prior from a 2x1 state vector z0 with a hardcoded timestamp.

    Parameters:
        z0: np.ndarray of shape (2,1) containing [x, y]
        vx0, vy0, w0: optional initial velocities and turn rate
    """
    
    #NOTE MEASUREMENT HAS TO BE 2X1 ARRAY
    x0 = z0[0,0] # i dont know why the measurements are a 2d array for 2 values -> it has to do with the stone soup implementation
    y0 = z0[1,0]

    #NOTE: IF CONSTANT VELOCITY SET w0 AS 0
    state_vec = np.array([[x0],
                        [vx0],
                        [y0],
                        [vy0],
                        [w0]])
    # large uncertainty on velocities and turn rate
    covar = np.diag([1.0, 10, 1.0, 10, 10.0])
    prior = GaussianState(state_vec, covar, timestamp=datetime(2025, 9, 12, 12, 0, 0))
    return prior      




    ##OPTIONAL MODELS
    #SINGER MODEL INSTED OF CONSTANT ACCELERATION -> UNNESECCARY
      # CONSTANT_ACCELERATION= { #PRIOR DEPENDS ON TRANSITION MODEL, MEASUREMENT MODEL DEPENDS ON PRIOR
    #     "trans_mod": CombinedLinearGaussianTransitionModel([
    #                                     ConstantAcceleration(0.05),  # Placeholder for acceleration model noise
    #                                     ConstantAcceleration(0.05)]),   # Placeholder for acceleration model noise  
                                        
    #     "prior": GaussianState([0, 0, 0, 0, 0, 0], np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), timestamp=start_time),
    #     "meas_mod": LinearGaussian(
    #                                     ndim_state=6,        
    #                                     mapping=(0, 3),      #TODO: CHECK IF POSITIONS ARE RIGHT measure x (0) and y (3) -> only position  
    #                                     noise_covar=np.array([[1e6, 0], #change to 3x3 if i insert velocity
    #                                                         [0, 1e6]])
    #                                 ),
    # }

   