import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuscenes.nuscenes import NuScenes
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
import numpy as np
from database import *
from datetime import timedelta, datetime
import plotly.io as pio
pio.renderers.default = "browser"
import matplotlib.pyplot as plt
import utils
from models.cartesian_models import Models as CartesianModels


start_time = datetime(2025, 9, 12, 12, 0, 0)  #fixed value

def prepare_movements(movements, model_data):
    step=0.5
    #convert movement dictionaries into ground truths (real state) and detections/measurements
    #measurement have added noise zk=Hkxk+vk
    detections, gt_states = [],[]
    
    R = model_data.value["meas_mod"].noise_covar 

    for i, mov in enumerate(movements):
         #if model_data == Models.CONSTANT_VELOCITY: -> maybe i will need rotation and velocity later and change the measurement data
        position_vector_2d = str_to_array(mov["translation"])[:2].reshape(-1, 1)
        
       
        gt_state = GroundTruthState( 
            state_vector=np.concatenate([
                position_vector_2d,
                np.array([[utils.quaternion_to_yaw(str_to_array(mov["rotation"]))]]),
                str_to_array(mov["velocity"])[:2].reshape(-1, 1)
            ], axis=0),
            timestamp=start_time + timedelta(seconds=i * step)
        )
        gt_states.append(gt_state)

        if R is not None: # Noisy measurement
            noisy_state = make_noisy(position_vector_2d, R)
        else:
            noisy_state = position_vector_2d

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
    aposteriori_track = Track()       # filtered/updated (posterior) track
    apriori_track = Track()  # predicted (prior) track

    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)  # Predict
        apriori_track.append(prediction)  
        
        hypothesis = SingleHypothesis(prediction, measurement) #measurement+noise
        post = updater.update(hypothesis)  # Update using measurement
        aposteriori_track.append(post)  
        prior = post  # swap for next step

    return apriori_track, aposteriori_track




if __name__== "__main__":
    instances = get_all_instances()    
    selected_instance = instances[0]
    movements = get_movements_by_instance(selected_instance["token"])
    #change according to tracked instance
    type = "COORDINATED_TURN"

    if type == "CONSTANT_VELOCITY":
        model_data = CartesianModels.CONSTANT_VELOCITY
        #linear kalman selected
        predictor = KalmanPredictor(model_data.value["trans_mod"])
        updater = KalmanUpdater(model_data.value["meas_mod"])
        gt_path, measurements = prepare_movements(movements, model_data)

    if type == "COORDINATED_TURN":
        model_data = CartesianModels.COORDINATED_TURN
        #UKF
        predictor = UnscentedKalmanPredictor(model_data.value["trans_mod"])
        updater = UnscentedKalmanUpdater(model_data.value["meas_mod"])
        gt_path, measurements = prepare_movements(movements, model_data) #for now the same

    apriori_track, aposteriori_track = kalman(measurements, model_data, predictor, updater) #plots priori aposteriori and gt
    #TEST
    for track in aposteriori_track:
        print(track.state_vector)
    
    data = utils.extract_state_data(gt_path, aposteriori_track) #extract data for plotting

    #combined plots of all individual metrics -> individual plot for each metric can also be called
    utils.plot_combined_tracks(data)

    #x and y 2D plot
    utils.plot_tracks_with_groundtruth(measurements, gt_path, apriori_track, aposteriori_track)

    plt.show()  


        