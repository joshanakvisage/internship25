from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
import numpy as np
from experiments.models.cartesian_models import Models as CartesianModels
from experiments.models.cartesian_models import generate_cartesian_prior_from_mes
from experiments.models.polar_models import PolarModels, generate_polar_prior_from_mes 
from experiments import utils #top level experiments utils
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis
from datetime import timedelta
import plotly.graph_objects as go


# def generate_perfect_trajectory_np(model_entry, first_point: np.ndarray, n_steps: int):
#     """
#     Generate a perfect trajectory as a NumPy array of state vectors (rows) 
#     and a list of Detection objects with noisy measurements.
#     Generated from generate_perfect_trajectory_positions by chatgpt
#     """
#     dt = 0.5
#     trans_model = model_entry.value["trans_mod"]
#     R = model_entry.value["meas_mod"].noise_covar    

#     # Initial state
#     current_state = generate_cartesian_prior_from_mes(first_point)

#     # allocate array for state vectors (n_steps × n_state_dim)
#     state_dim = current_state.state_vector.shape[0]
#     states_array = np.zeros((n_steps, state_dim))

#     # fill first row
#     states_array[0, :] = current_state.state_vector.ravel()

#     # detections list
#     detections = [
#         Detection(
#             state_vector=utils.make_noisy(
#                 current_state.state_vector[[0, 2], :],  # x,y only
#                 R
#             ),
#             timestamp=current_state.timestamp,
#         )
#     ]

#     # propagate
#     for i in range(1, n_steps):
#         next_state_vector = trans_model.function(
#             current_state,
#             noise=False,
#             time_interval=timedelta(seconds=dt),
#         )

#         current_state = GaussianState(
#             state_vector=next_state_vector,
#             covar=current_state.covar.copy(),
#             timestamp=current_state.timestamp + timedelta(seconds=dt),
#         )

#         # save full state vector row
#         states_array[i, :] = current_state.state_vector.ravel()

#         # add detection (noisy x,y)
#         xy = current_state.state_vector[[0, 2], :]  # x,y only
#         detections.append(
#             Detection(
#                 state_vector=utils.make_noisy(xy, R),
#                 timestamp=current_state.timestamp,
#             )
#         )

#     return states_array, detections

def generate_perfect_trajectory_np(model_entry, first_point: np.ndarray, n_steps: int):
    """
    Generate a perfect trajectory as a NumPy array of state vectors (rows) 
    and a list of Detection objects with noisy measurements.
    Works for both Cartesian and Polar CT models.
    """
    dt = 0.5
    trans_model = model_entry.value["trans_mod"]
    R = model_entry.value["meas_mod"].noise_covar    

    # Initial state: pick correct generator based on model
    if "POLAR" in model_entry.name:  
        current_state = generate_polar_prior_from_mes(first_point, v=10., h=-2, ω=0.15)
    else:
        current_state = generate_cartesian_prior_from_mes(first_point, vx0 = -10., vy0= -2., w0=0.2)

    # allocate array for state vectors (n_steps × n_state_dim)
    state_dim = current_state.state_vector.shape[0]
    states_array = np.zeros((n_steps, state_dim))

    # fill first row
    states_array[0, :] = current_state.state_vector.ravel()

    # Use measurement mapping from the model definition
    meas_indices = model_entry.value["meas_mod"].mapping

    # detections list
    detections = [
        Detection(
            state_vector=utils.make_noisy(
                current_state.state_vector[meas_indices, :],  
                R
            ),
            timestamp=current_state.timestamp,
        )
    ]

    # propagate
    for i in range(1, n_steps):
        next_state_vector = trans_model.function(
            current_state,
            noise=False, #NOTE: IMPORTANT 
            time_interval=timedelta(seconds=dt),
        )

        current_state = GaussianState(
            state_vector=next_state_vector,
            covar=current_state.covar.copy(),
            timestamp=current_state.timestamp + timedelta(seconds=dt),
        )

        # save full state vector row
        states_array[i, :] = current_state.state_vector.ravel()

        # add detection (noisy projection using mapping)
        detections.append(
            Detection(
                state_vector=utils.make_noisy(
                    current_state.state_vector[meas_indices, :],
                    R
                ),
                timestamp=current_state.timestamp,
            )
        )

    return states_array, detections



def plot_states_xy(transition_model, states_array, color="blue"):
    
    # extract x and y (column 0 and column 2)
    if "POLAR" in transition_model.name:  
        x_vals = states_array[:, 0]
        y_vals = states_array[:, 1]
    else:
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




def plot_mons(mons_values, title="MoN values over path steps"):
    """
    Plots the MoN values over the path steps.
    
    Parameters:
    mons_values (list or np.ndarray): List or array of MoN values
    title (str): Title of the plot
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=mons_values,
        mode='lines+markers',
        name='MoN',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Path Step (Index)',
        yaxis_title='MoN Value',
        template='plotly_white',
        width=800,
        height=600
    )

    fig.show()


#TODO: ZA GT PATH VERZIJU MONA MORAM DODAT W0 I TO STA CE SE PREUZET OD GT-PATHA TOG
def kalman(measurements: list, model_variables, predictor, updater):
    if model_variables == CartesianModels.COORDINATED_TURN:
        prior = generate_cartesian_prior_from_mes(measurements[0].state_vector,  vx0 = -10., vy0= -2., w0=0.2)
    elif model_variables == PolarModels.POLAR_COORDINATED_TURN:
        prior = generate_polar_prior_from_mes(measurements[0].state_vector, v=10., h=-2, ω=0.15)
    else:
        raise ValueError("Unknown model type passed to kalman()")
    

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




def test_affine_mapping(u, f, pos_indices, tol=1e-10):
    """
    Check if the mapping u -> f is approximately affine on the selected positions.
    u: shape (n_paths, state_dim)
    f: shape (n_paths, state_dim)
    pos_indices: list of position indices (e.g., [0,2] for x,y in Cartesian)
    tol: relative residual threshold to consider mapping affine
    """
    # select only the positions
    u_pos = u[:, pos_indices]
    f_pos = f[:, pos_indices]

    # augment u with ones for bias term
    U = np.hstack([u_pos, np.ones((u_pos.shape[0], 1))])

    # solve least-squares: U @ W = f_pos
    W, *_ = np.linalg.lstsq(U, f_pos, rcond=None)

    # predicted f
    f_hat = U @ W

    # relative residual
    res_norm = np.linalg.norm(f_pos - f_hat) / np.linalg.norm(f_pos)

    if res_norm < tol:
        print(f"Mapping is effectively affine, residual {res_norm:.3e}")
    else:
        print(f" Mapping is nonlinear, residual {res_norm:.3e}")

    return res_norm, W, f_hat