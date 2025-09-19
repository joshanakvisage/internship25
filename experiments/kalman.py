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
from models.polar_models import PolarModels 

start_time = datetime(2025, 9, 12, 12, 0, 0)  #fixed value


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



#14 je ODLICAN ZA SADA
if __name__== "__main__":
    instances = get_all_instances()    
    selected_instance = instances[4]
    print(selected_instance["token"])
    movements = get_movements_by_instance(selected_instance["token"])
    #change according to tracked instance
    type = "COORDINATED_TURN"
    #type = "POLAR_COORDINATED_TURN"
    if type == "CONSTANT_VELOCITY":
        model_data = CartesianModels.CONSTANT_VELOCITY
        #linear kalman selected
        predictor = KalmanPredictor(model_data.value["trans_mod"])
        updater = KalmanUpdater(model_data.value["meas_mod"])
        gt_path, measurements = utils.prepare_movements(movements, model_data) 

    if type == "COORDINATED_TURN":
        model_data = CartesianModels.COORDINATED_TURN
        #UKF
        predictor = UnscentedKalmanPredictor(model_data.value["trans_mod"])
        updater = UnscentedKalmanUpdater(model_data.value["meas_mod"])
        #gt_path, measurements = prepare_movements(movements, model_data)
        gt_path, measurements = utils.prepare_movements(movements, model_data, start_time, step=0.5, interpolate_points=True)

    if type == "POLAR_COORDINATED_TURN":
        model_data = PolarModels.POLAR_COORDINATED_TURN
        #UKF
        predictor = UnscentedKalmanPredictor(model_data.value["trans_mod"])
        updater = UnscentedKalmanUpdater(model_data.value["meas_mod"])
        gt_path, measurements = utils.prepare_movements(movements, model_data) #for now the same, can change it if i want to add v1 or v2


    apriori_track, aposteriori_track = kalman(measurements, model_data, predictor, updater) #plots priori aposteriori and gt
    #TEST
    i=0
    print(len(aposteriori_track))
    for track in gt_path:
         print(i)
         print(track.state_vector)
         i=i+1
    

    data = utils.extract_state_data(gt_path, aposteriori_track, type) #extract data for plotting

    #combined plots of all individual metrics -> individual plot for each metric can also be called
    utils.plot_combined_tracks(data)

    #x and y 2D plot
    utils.plot_tracks_with_groundtruth(measurements, gt_path, apriori_track, aposteriori_track)

    plt.show()  


        