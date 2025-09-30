from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
import numpy as np
from experiments.models.cartesian_models import Models as CartesianModels
from experiments.models.cartesian_models import generate_cartesian_prior_from_mes
from experiments.models.polar_models import PolarModels 
from experiments import utils #top level experiments utils
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis
from datetime import timedelta
import plotly.graph_objects as go

def generate_perfect_trajectory_positions(model_entry, first_point: np.ndarray , n_steps: int):
    
    dt = 0.5
    trans_model, R = model_entry.value["trans_mod"], model_entry.value["meas_mod"].noise_covar    
    prior_state = generate_cartesian_prior_from_mes(first_point)
    
    current_state = prior_state

    positions = np.zeros((n_steps, 2))
    positions[0, 0], positions[0, 1]  = current_state.state_vector[0, 0], current_state.state_vector[2, 0]  # first x and y
    

    gt_states = [GroundTruthState(current_state.state_vector, current_state.timestamp)]
    detections = [Detection(state_vector=utils.make_noisy(positions[0, :].reshape(-1, 1), R),
                                timestamp=current_state.timestamp)]
    
   
    
    for i in range(1, n_steps):
        
        next_state_vector = trans_model.function(
            current_state,
            noise=False,
            time_interval=timedelta(seconds=dt)
        )
        
        
        current_state = GaussianState(
        state_vector=next_state_vector,  # use the StateVector object directly
        covar=current_state.covar.copy(),
        timestamp=current_state.timestamp + timedelta(seconds=dt)
        )

        positions[i, 0], positions[i, 1] = current_state.state_vector[0, 0], current_state.state_vector[2, 0]  # x and y
   
        
        gt_states.append(GroundTruthState(current_state.state_vector, current_state.timestamp))
        detections.append(Detection(state_vector=utils.make_noisy(positions[i, :].reshape(-1, 1), R),
                                timestamp=current_state.timestamp))


    gt_path = GroundTruthPath(gt_states)
    return gt_path, detections, positions

def generate_perfect_trajectory_np(model_entry, first_point: np.ndarray, n_steps: int):
    """
    Generate a perfect trajectory as a NumPy array of state vectors (rows) 
    and a list of Detection objects with noisy measurements.
    Generated from generate_perfect_trajectory_positions by chatgpt
    """
    dt = 0.5
    trans_model = model_entry.value["trans_mod"]
    R = model_entry.value["meas_mod"].noise_covar    

    # Initial state
    current_state = generate_cartesian_prior_from_mes(first_point)

    # allocate array for state vectors (n_steps × n_state_dim)
    state_dim = current_state.state_vector.shape[0]
    states_array = np.zeros((n_steps, state_dim))

    # fill first row
    states_array[0, :] = current_state.state_vector.ravel()

    # detections list
    detections = [
        Detection(
            state_vector=utils.make_noisy(
                current_state.state_vector[[0, 2], :],  # x,y only
                R
            ),
            timestamp=current_state.timestamp,
        )
    ]

    # propagate
    for i in range(1, n_steps):
        next_state_vector = trans_model.function(
            current_state,
            noise=False,
            time_interval=timedelta(seconds=dt),
        )

        current_state = GaussianState(
            state_vector=next_state_vector,
            covar=current_state.covar.copy(),
            timestamp=current_state.timestamp + timedelta(seconds=dt),
        )

        # save full state vector row
        states_array[i, :] = current_state.state_vector.ravel()

        # add detection (noisy x,y)
        xy = current_state.state_vector[[0, 2], :]  # x,y only
        detections.append(
            Detection(
                state_vector=utils.make_noisy(xy, R),
                timestamp=current_state.timestamp,
            )
        )

    return states_array, detections


def plot_states_xy(states_array, color="blue"):
    
    # extract x and y (column 0 and column 2)
    x_vals = states_array[:, 0]
    y_vals = states_array[:, 2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines+markers',
        line=dict(color=color),
        name='Trajectory'
    ))
    fig.update_layout(
        title="Generated Trajectory",
        xaxis_title="X",
        yaxis_title="Y",
        width=800,
        height=600
    )
    fig.show()


#TODO: ZA GT PATH VERZIJU MONA MORAM DODAT W0 I TO STA CE SE PREUZET OD GT-PATHA TOG
def kalman(measurements: list, model_variables, predictor, updater):
    if model_variables ==  CartesianModels.COORDINATED_TURN:
        prior = generate_cartesian_prior_from_mes(measurements[0].state_vector) #NOTE: add vx0, vy0, w0 as parameters for prior state vector SET W0 AS 0 FOR CV
    else:
        prior = None #in case polar
    aposteriori_track = Track()       # filtered/updated (posterior) track
    apriori_track = Track()  # predicted (prior) track
    posterior_states = []

    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)  # PREDICTION
        apriori_track.append(prediction)  
        hypothesis = SingleHypothesis(prediction, measurement) #measurement+noise
        post = updater.update(hypothesis)  # Update using measurement
        aposteriori_track.append(post)  
        prior = post  # swap for next step
        posterior_states.append(post.state_vector.ravel())
        
    posterior_array = np.vstack(posterior_states)
    return apriori_track, aposteriori_track, posterior_array
