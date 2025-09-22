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
from stonesoup.types.state import GaussianState

start_time = datetime(2025, 9, 12, 12, 0, 0)  #fixed value

def generate_prior_from_mes(measurements: list, model_variables):
    first_meas = measurements[0]
    z0 = first_meas.state_vector  # shape (2,1): [x, y]
    x0 = z0[0,0] # i dont know why the measurements are a 2d array for 2 values -> it has to do with the stone soup implementation
    y0 = z0[1,0]
    # guess zero velocity and zero turn rate
    vx0, vy0, w0 = 0., 0., 0.
    #change this depending on the transition model (look up from models)
    state_vec = np.array([[x0],
                        [vx0],
                        [y0],
                        [vy0],
                        [w0]])
    # large uncertainty on velocities and turn rate
    covar = np.diag([1.0, 100.0, 1.0, 100.0, 10.0])
    prior = GaussianState(state_vec, covar, timestamp=first_meas.timestamp)
    return prior


def kalman(measurements: list, model_variables, predictor, updater):
    #chose if first prior is from first measurement data or if prior is hardcoded as 0,0
    prior = generate_prior_from_mes(measurements, model_variables) #decrease transition noise and increase measurement noise
    #prior = model_variables.value["prior"] #NOTE: if prior is hardcoded from model class decrease measurement noise and increase transition noise
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
    # print(len(aposteriori_track))
    # for track in gt_path:
    #      print(i)
    #      print(track.state_vector)
    #      i=i+1
    

    data = utils.extract_state_data(gt_path, aposteriori_track, type) #extract data for plotting

    #combined plots of all individual metrics -> individual plot for each metric can also be called
    utils.plot_combined_tracks(data)

    #x and y 2D plot
    utils.plot_tracks_with_groundtruth(measurements, gt_path, apriori_track, aposteriori_track)

    plt.show()  


        