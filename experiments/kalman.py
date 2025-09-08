import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuscenes.nuscenes import NuScenes
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
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


start_time = datetime.now() 

#Here the appropriate model will be selected
#Returns a prior, translation model and measurement model
class Models(Enum):
    Linear = 0

# Convert dict to Detection for stonesoup
def prepare_movements(movements, step=0.5):
    detections = []
    for i, mov in enumerate(movements):
        position_3d = str_to_array(mov["translation"])
        position_2d = position_3d[:2] # Discard z (only keep x, y)
        # Create Detection object 
        detection = Detection(
            state_vector= position_2d.reshape(-1, 1),
            timestamp=start_time + timedelta(seconds=i * step)
        )
        detections.append(detection)

    return detections


def kalman(measurements: list):
    q_x = 0.05 #process noise? -> The target is assumed to move with (nearly) constant velocity, where target acceleration is modelled as white noise.
    q_y = 0.05
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])
    
    measurement_model = LinearGaussian(
    ndim_state=4,  # Number of state fectors
    mapping=(0, 2),  # mapping so the measurement index fits the predicted state vector H_x*x_k-1
    noise_covar=np.array([[0.1, 0], # Covariance matrix for Gaussian PDF
                          [0, 0.1]])
    ) 
    
    # print("Transition model")
    # print(f"{transition_model.matrix(time_interval=timedelta(seconds=0.5))}")
    # print("Measurement model")
    # print(f"{measurement_model.matrix()}")
    
    

    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    #TODO: SET FIRST PRIOR THE SAME AS MEASUREMENT? OTHERWISE LARGE DISCREPENCY
    prior = GaussianState([[0], [1], [0], [1]], np.diag([0.5, 0.5, 0.5, 0.5]), timestamp=start_time)


    track = Track()       # filtered (posterior) track
    pred_track = Track()  # predicted (prior) track


    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)  # Predict
        pred_track.append(prediction)  

        hypothesis = SingleHypothesis(prediction, measurement)
        post = updater.update(hypothesis)  # Update using measurement
        track.append(post)  


    prior = post  # swap for next step
    return pred_track, track




if __name__== "__main__":
    instances = get_all_instances()   
    first_instance = instances[0]
    movements = get_movements_by_instance(first_instance["token"])

    detections = prepare_movements(movements)
    #print(detections[12])
    #print(movements[12]["translation"])
    pred_track, track = kalman(detections)

    timestamps = [detection.timestamp for detection in detections]

    
    plotter = AnimatedPlotterly(timestamps, tail_length=0.3)
    plotter.plot_tracks(pred_track, [0, 2], uncertainty=True, line=dict(color="orange"))
    plotter.plot_tracks(track, [0, 2], uncertainty=True)
    plotter.fig.show()

    