
from stonesoup.models.measurement.linear import LinearGaussian
from enum import Enum
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity, ConstantAcceleration
import numpy as np
from stonesoup.types.state import GaussianState
from stonesoup.models.transition.nonlinear import ConstantTurn 
from datetime import timedelta, datetime
from stonesoup.models.transition.nonlinear import NonLinearTransitionModel
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange


start_time = datetime(2025, 9, 12, 12, 0, 0) 

class PolarCoordinatedTurn(NonLinearTransitionModel):
    ndim_state = 5  # x, y, v, h, omega


    def function(self, state, noise=None, time_interval=None, **kwargs):
        x1, x2, v, h, omega = state.state_vector.flatten()
        T = time_interval.total_seconds() if time_interval else 1.0 #total_seconds does not round up?

        if abs(omega) < 1e-6:  # straight-line, avoids divison by 0
            x1_new = x1 + v*T*np.cos(h)
            x2_new = x2 + v*T*np.sin(h)
        else:
            x1_new = x1 + (2*v/omega)*np.sin(omega*T/2)*np.cos(h+omega*T/2)
            x2_new = x2 + (2*v/omega)*np.sin(omega*T/2)*np.sin(h+omega*T/2)

        v_new = v
        h_new = h + omega*T
        omega_new = omega

        new_state = np.array([[x1_new], [x2_new], [v_new], [h_new], [omega_new]])

        if noise is not None:
            new_state += noise

        return new_state
    
#prior/state vector should be according to the paper EKF/UKF maneuvering target tracking using coordinated models with polar/cartesian velocities
# x1:Cartesian x-position
# x2:Cartesian y-position
# v:speed magnitude (m/s) (the norm of velocity)
# h:heading angle (radians)
# ω:turn rate (rad/s)
class PolarModels(Enum):
    POLAR_COORDINATED_TURN = {
        "trans_mod": PolarCoordinatedTurn(),
        "prior": GaussianState(
            state_vector=np.array([[1.0], [0.0], [0.0], [0.1], [0.0]]),
            covar=np.diag([0.5, 0.1, 0.5, 0.1, 0.01]),
            timestamp=start_time
        ),
        "meas_mod": LinearGaussian(
            ndim_state=5,
            mapping=(0, 1),  # x and y positions in state vector 
            noise_covar=np.diag([0.1, 0.1])
        )
    }

