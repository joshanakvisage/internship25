
from stonesoup.models.measurement.linear import LinearGaussian
from enum import Enum
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity, ConstantAcceleration
import numpy as np
from stonesoup.types.state import GaussianState 
from datetime import timedelta, datetime
from stonesoup.models.transition.nonlinear import GaussianTransitionModel
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange #not needed for now
from stonesoup.base import Property
from stonesoup.types.state import StateVector, StateVectors


start_time = datetime(2025, 9, 12, 12, 0, 0) 


class PolarCoordinatedTurn(GaussianTransitionModel):
    linear_noise: float = Property(default=0.1)
    turn_noise: float = Property(default=0.01)
    
    @property
    def ndim_state(self):
        return 5


    #TODO: FIX THIS
    
    def covar(self, time_interval=None, **kwargs):
        # Compute covariance on the fly from Properties
        return np.diag([self.linear_noise, self.linear_noise, 0.01, 0.01, self.turn_noise])
    #if the argument is a 1D state vector, but UKF takes in sigma ponts (5vectors)
    # def function(self, state, noise=None, time_interval=None, **kwargs):
    #     print(state.state_vector)
    #     sv = np.atleast_2d(state.state_vector)
    #     if sv.shape[0] == 1 and sv.shape[1] == 5:
    #         sv = sv.T  # make (5,1)

    #     x1, x2, v, h, omega = sv.flatten()
    #     T = time_interval.total_seconds() if time_interval else 1.0

    #     if abs(omega) < 1e-6:  # straight-line
    #         x1_new = x1 + v*T*np.cos(h)
    #         x2_new = x2 + v*T*np.sin(h)
    #     else:
    #         x1_new = x1 + (2*v/omega)*np.sin(omega*T/2)*np.cos(h+omega*T/2)
    #         x2_new = x2 + (2*v/omega)*np.sin(omega*T/2)*np.sin(h+omega*T/2)

    #     v_new = v
    #     h_new = h + omega*T
    #     omega_new = omega

    #     new_state = np.array([[x1_new], [x2_new], [v_new], [h_new], [omega_new]])

    #     if noise is not None:
    #         new_state += noise
    #     return new_state


    
    def function(self, state, noise=False, **kwargs) -> StateVector:
        #UKF adaptable Polar Coordinated Turn function analogous to Cartesian CT.
        #Expects state.state_vector of shape (5, n_sigma) > each row state vector variable, each column sigma point
        #according to the function in EKF/UKF maneuvering target teacking using CT with Polar/cartesian velocity
        
        dt = kwargs["time_interval"].total_seconds()
        sv1 = state.state_vector  # matrix shape 
        x = sv1[0, :]
        y = sv1[1, :]
        v = sv1[2, :]
        heading = sv1[3, :]
        omega = sv1[4, :]

        # Avoid divide by zero for small turn rates
        omega_safe = omega.copy()
        omega_safe[np.abs(omega_safe) < 1e-6] = 1e-12

        dAngle = omega * dt
       
        #vectorized conditional, if omega is less than the threshold calculate by constant velocity, otherwise polar
        x_new = np.where(np.abs(omega) < 1e-6,
                         x + v * dt * np.cos(heading),
                         x + (2*v/omega_safe) * np.sin(omega*dt/2) * np.cos(heading + omega*dt/2))
        #vectorized conditional, if omega is less than the threshold calculate by constant velocity, otherwise polar
        y_new = np.where(np.abs(omega) < 1e-6,
                         y + v * dt * np.sin(heading),
                         y + (2*v/omega_safe) * np.sin(omega*dt/2) * np.sin(heading + omega*dt/2))

        # Heading and velocity updates
        v_new = v
        heading_new = heading + dAngle
        omega_new = omega

        #new sigma points matrix
        sv2 = StateVectors([x_new, y_new, v_new, heading_new, omega_new])

        # Add process noise if requested
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0
        #result of function is a matrix of sigma point values - later on all the sigma vector wills be averaged by weights (internally implemented in UKF)
        return sv2 + noise 

    
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
            covar=np.diag([0.5, 0.1, 0.5, 0.1, 0.01]), #P 
            timestamp=start_time
        ),
        "meas_mod": LinearGaussian(
            ndim_state=5,
            mapping=(0, 1),  # x and y positions in state vector 
            noise_covar=np.diag([0.1, 0.1])
        )
    }

