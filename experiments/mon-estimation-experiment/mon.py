import numpy as np
from experiments.models.cartesian_models import Models as CartesianModels
from experiments.models.polar_models import PolarModels 
from experiments import utils #top level experiments utils
from .utils import add #mon-estimation-experiment utils
from stonesoup.types.state import GaussianState
from datetime import timedelta
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from numpy.linalg import pinv
def plot_positions_numpy(positions, color="blue"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode='lines+markers',
        line=dict(color=color),
        name='Trajectory'
    ))
    fig.update_layout(
        title="Generated Trajectory",
        xaxis_title="X",
        yaxis_title="Y",
        width=800, height=600
    )
    fig.show()

def generate_perfect_trajectory_positions(model_entry, n_steps: int):
    
    dt = 0.5
    trans_model = model_entry.value["trans_mod"]
    prior_state = model_entry.value["prior"]
    
    current_state = prior_state
    states = [current_state]  # include initial prior
    
    
    positions = np.zeros((n_steps, 2))
    
    positions[0, 0] = current_state.state_vector[0, 0]  # x
    positions[0, 1] = current_state.state_vector[2, 0]  # y
    
    for i in range(1, n_steps):
        
        next_state_vector = trans_model.function(
            current_state,
            noise=False,
            time_interval=timedelta(seconds=dt)
        )
        
        
        next_state = GaussianState(
        state_vector=next_state_vector,  # use the StateVector object directly
        covar=current_state.covar.copy(),
        timestamp=current_state.timestamp + timedelta(seconds=dt)
        )

        states.append(next_state)
        current_state = next_state
        
        
        positions[i, 0] = next_state.state_vector[0, 0]  # x
        positions[i, 1] = next_state.state_vector[2, 0]  # y
    
    return states, positions

def extract_transition_data(states, transition_model, dt):
    #Extracts state vectors (U) and their transitioned state vectors (F)
    #from a list of Stone Soup GaussianState objects using a transition model.
    U = np.hstack([s.state_vector for s in states[:-1]])  # exclude last
    
    # Outputs: next state predicted by model
    F = np.hstack([
        transition_model.function(
            s, noise=False,time_interval= timedelta(seconds=dt) 
        ) for s in states[:-1]
    ])
    
    return U, F

def compute_covariances(U, F):
    #Compute covariance matrices Σ_uu, Σ_ff, Σ_fu needed for MoN.
    U_mean = np.mean(U, axis=1, keepdims=True)
    F_mean = np.mean(F, axis=1, keepdims=True)
    
    U_centered = U - U_mean
    F_centered = F - F_mean
    
    n_samples = U.shape[1]
    
    Sigma_uu = (U_centered @ U_centered.T) / (n_samples - 1)
    Sigma_ff = (F_centered @ F_centered.T) / (n_samples - 1)
    Sigma_fu = (F_centered @ U_centered.T) / (n_samples - 1)
    
    return Sigma_uu, Sigma_ff, Sigma_fu, U_mean, F_mean

def compute_mon(Sigma_uu, Sigma_ff, Sigma_fu):
    #Compute the unitless Measure of Nonlinearity M.
    #M = tr(Σ_ff - Σ_fu Σ_uu^-1 Σ_fu^T)
    #inv_Sigma_uu = pinv(Sigma_uu)
    inv_Sigma_uu = np.linalg.inv(Sigma_uu)
    M = np.trace(Sigma_ff - Sigma_fu @ inv_Sigma_uu @ Sigma_fu.T)
    return M
def measure_of_nonlinearity(states, transition_model, dt):
    #Compute the Measure of Nonlinearity (MoN) for a transition model
    #given a list of GaussianState objects.
    U, F = extract_transition_data(states, transition_model, dt)
    Sigma_uu, Sigma_ff, Sigma_fu, U_mean, F_mean = compute_covariances(U, F)
    
    M = compute_mon(Sigma_uu, Sigma_ff, Sigma_fu)
    return M


if __name__=="__main__":
    states , positions = generate_perfect_trajectory_positions(CartesianModels.COORDINATED_TURN, 50)
    plot_positions_numpy(positions)
    #M= measure_of_nonlinearity(states, CartesianModels.COORDINATED_TURN.value["trans_mod"], 0.5)
    #print(M)



