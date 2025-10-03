import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from database import *
from datetime import timedelta, datetime
import plotly.io as pio
pio.renderers.default = "browser"
import matplotlib.pyplot as plt
import utils
from models.cartesian_models import Models as CartesianModels
from models.cartesian_models import generate_cartesian_prior_from_mes
from models.polar_models import PolarModels 
import numpy as np

start_time = datetime(2025, 9, 12, 12, 0, 0)  #fixed value


def kalman(measurements: list, model_variables, predictor, updater):
    prior = generate_cartesian_prior_from_mes(measurements[0].state_vector, vx0=-10, vy0=-5, w0=0.2) #NOTE: add vx0, vy0, w0 as parameters SET W0 AS 0 FOR CV
    aposteriori_track = Track()       # filtered/updated (posterior) track
    apriori_track = Track()  # predicted (prior) track
    posterior_states = []

    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)  # PREDICTION
        apriori_track.append(prediction)  
        #COMMENT THIS TOSEE THE TRANS FUNCTION
        hypothesis = SingleHypothesis(prediction, measurement) #measurement+noise
        post = updater.update(hypothesis)  # Update using measurement
        aposteriori_track.append(post)  
        prior = post  # swap for next step
        posterior_states.append(post.state_vector.ravel())
        #aposteriori_track.append(prediction) #NOTE: UNCOMMENT THIS TO SEE TRANSITION FUNCTION
        
    posterior_array = np.vstack(posterior_states)
    return apriori_track, aposteriori_track, posterior_array
    #return apriori_track, aposteriori_track, 0



#14 je ODLICAN ZA SADA
if __name__== "__main__":
    instances = get_all_instances()    
    selected_instance = instances[6]
    print(selected_instance["token"])
    movements = get_movements_by_instance(selected_instance["token"])
    #change according to tracked instance
    #type = "CONSTANT_VELOCITY"
    type = "COORDINATED_TURN"
    #type = "POLAR_COORDINATED_TURN"
    if type == "CONSTANT_VELOCITY":
        model_data = CartesianModels.CONSTANT_VELOCITY
        #linear kalman selected
        predictor = KalmanPredictor(model_data.value["trans_mod"])
        updater = KalmanUpdater(model_data.value["meas_mod"])
        gt_path, measurements = utils.prepare_movements(movements, model_data) 
    else:
        if type == "COORDINATED_TURN":
            model_data = CartesianModels.COORDINATED_TURN
        if type == "POLAR_COORDINATED_TURN":
            model_data = PolarModels.POLAR_COORDINATED_TURN
            #UKF
        predictor = UnscentedKalmanPredictor(model_data.value["trans_mod"])
        updater = UnscentedKalmanUpdater(model_data.value["meas_mod"])
        #measurements are array of detections
        gt_path, measurements = utils.prepare_movements(movements, model_data, start_time, step=0.5, interpolate_points=False)
        


    apriori_track, aposteriori_track, posterior_array = kalman(measurements, model_data, predictor, updater) #plots priori aposteriori and gt
    #TEST
    i=0
    # print(len(aposteriori_track))
    #print(posterior_array[-1][0])
    # for track in aposteriori_track:
    #       print(i)
    #       print(track) 
    #       i=i+1
    

    data = utils.extract_state_data(gt_path, aposteriori_track, type) #extract data for plotting

    #combined plots of all individual metrics -> individual plot for each metric can also be called
    utils.plot_combined_tracks(data)

    #x and y 2D plots
    utils.plot_tracks_with_groundtruth(measurements, gt_path, apriori_track, aposteriori_track)

    plt.show()  


        