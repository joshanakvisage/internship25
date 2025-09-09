import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuscenes.nuscenes import NuScenes
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity, ConstantAcceleration
from stonesoup.models.transition.nonlinear import ConstantTurn          
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater

from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian
import numpy as np
from database import *
import plotly.io as pio
from datetime import timedelta, datetime
from stonesoup.types.track import Track
pio.renderers.default = "browser"
from enum import Enum

from stonesoup.plotter import AnimatedPlotterly

#ANALIZIRAO BIH KOORDINATNI TURN  CONST VEL SA KARTEZIJEVIM I POLARNIM SUSTAVIMA
#ISTI MODEL S DRUGIM IZRAŽAVANJEM 



start_time = datetime.now() 

#Here the appropriate model will be selected
#Returns a translation model, a prior (compatable to the t mod.) and measurement model (compatable to the prior)
class Models(Enum):
    CONSTANT_VELOCITY = { #x_{k+1} = x_k + v_x T -> pedestrians, slow objects
        "trans_mod" : CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05 ),
                                                          ConstantVelocity(0.05 )]),
        "prior" : GaussianState([[0], [0], [0], [0]], np.diag([0.5, 0.5, 0.5, 0.5]), timestamp=start_time), #krivo init treba povećat
        "meas_mod" : LinearGaussian(
            ndim_state=4,  # Number of state vectors
            mapping=(0, 2),  # mapping so the measurement index fits the predicted state vector H_x*x_k-1
            noise_covar=np.array([[0.1, 0], 
                                [0, 0.1]]) #where R is your measurement noise covariance.
            ) 
    }
    CONSTANT_ACCELERATION= {
        "trans_mod": CombinedLinearGaussianTransitionModel([
                                        ConstantAcceleration(0.05),  # Placeholder for acceleration model noise
                                        ConstantAcceleration(0.05)]),   # Placeholder for acceleration model noise  
                                        
        "prior": GaussianState([0, 0, 0, 0, 0, 0], np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), timestamp=start_time),
        "meas_mod": LinearGaussian(
                                        ndim_state=6,        
                                        mapping=(0, 3),      # measure x (0) and y (3) -> only position 
                                        noise_covar=np.array([[0.1, 0], #change to 3x3 if i insert velocity
                                                            [0, 0.1]])
                                    ),
    }
    #SINGER MODEL INSTED OF CONSTANT ACCELERATION?
    COORDINATED_TURN = {
        "trans_mod": ConstantTurn(
            turn_noise_coeff=0.01,  # process noise for turn rate ω
            linear_noise_coeffs=np.array([0.05, 0.05, 0.01, 0.01, 0.01])  
            # noise for [x, y, v, heading, ω]
        ),
        "prior":GaussianState(
                state_vector=np.array([[0],   # x position
                                    [1],   # x velocity
                                    [0],   # y position
                                    [1],   # y velocity
                                    [0]]), # turn rate ω
                covar=np.diag([0.5, 0.5, 0.5, 0.5, 0.01]),  # Covariance for each state
                timestamp=start_time),
        "meas_mod": LinearGaussian(
            ndim_state=5,             # number of states in ConstantTurn
            mapping=(0, 2),           # we can measure x and y positions from sensors
            noise_covar=np.array([[0.1, 0], 
                                [0, 0.1]])  # measurement noise for x and y
        )
    },
    CONSTANT_TURN_ACCELERATION = {
        "trans_mod": 0,
        "prior":0,
        "meas_mod":0
    }



def prepare_movements(movements, model_data):
    step=0.5
    #convert movement dictionaries into ground truths (real state) and detections/measurements
    #measurement have added noise zk=Hkxk+vk
    detections, gt_states = [],[]
    
    R = model_data.value["meas_mod"].noise_covar 

    for i, mov in enumerate(movements):
        position_3d = str_to_array(mov["translation"])
        
        if model_data == Models.CONSTANT_VELOCITY:
            position_vector = position_3d[:2].reshape(-1, 1) # Only x, y, discard z and any rotation/velocity
        else:
            print("Not set yet") # for other models, you might want x, y, z, rotation, etc. - add move["rotation"]
        
        gt_state = GroundTruthState( 
            state_vector=np.concatenate([
                position_3d.reshape(-1, 1),
                str_to_array(mov["rotation"]).reshape(-1, 1),
                str_to_array(mov["velocity"]).reshape(-1, 1)
            ], axis=0),
            timestamp=start_time + timedelta(seconds=i * step)
        )
        gt_states.append(gt_state)

        if R is not None: # Noisy measurement
            noisy_state = make_noisy(position_vector, R)
        else:
            noisy_state = position_vector

        detection = Detection(
            state_vector=noisy_state,
            timestamp=start_time + timedelta(seconds=i * step)
        )
        detections.append(detection)

    gt_path = GroundTruthPath(gt_states)
    return gt_path, detections
    
def make_noisy(data, R):
    mean = np.zeros(R.shape[0])
    noise = np.random.multivariate_normal(mean, R).reshape(-1, 1)
    return data+noise


def kalman(measurements: list, model_variables, predictor, updater):
    
    prior = model_variables.value["prior"]

    track = Track()       # filtered (posterior) track
    pred_track = Track()  # predicted (prior) track


    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)  # Predict
        pred_track.append(prediction)  
        
        hypothesis = SingleHypothesis(prediction, measurement) #measurement+noise
        post = updater.update(hypothesis)  # Update using measurement
        track.append(post)  
        prior = post  # swap for next step

    return pred_track, track




if __name__== "__main__":
    instances = get_all_instances()  
    first_instance = instances[1]
    movements = get_movements_by_instance(first_instance["token"])
    
    #change according to tracked instance
    type = "CONSTANT_VELOCITY"

    if type == "CONSTANT_VELOCITY":
        model_data = Models.CONSTANT_VELOCITY
        #linear kalman selected
        predictor = KalmanPredictor(model_data.value["trans_mod"])
        updater = KalmanUpdater(model_data.value["meas_mod"])
        gt_path, measurements = prepare_movements(movements, model_data)

    if type == "COORDINATED_TURN":
        model_data = Models.COORDINATED_TURN
        predictor = UnscentedKalmanPredictor(model_data.value["trans_mod"])
        updater = UnscentedKalmanPredictor(model_data.valu["meas_mod"])
        gt_path, measurements = prepare_movements(movements, model_data) #for now the same

    pred_track, track = kalman(measurements, model_data, predictor, updater)

    timestamps = [detection.timestamp for detection in measurements]
    #compare ground truth velocity with predicted velocity
    plotter = AnimatedPlotterly(timestamps, tail_length=0.3)
    plotter.plot_tracks(pred_track, [0, 2], uncertainty=True, line=dict(color="orange"))
    plotter.plot_tracks(track, [0, 2], uncertainty=True)
    plotter.fig.show()

    