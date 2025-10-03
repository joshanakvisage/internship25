import numpy as np
from experiments.models.cartesian_models import Models as CartesianModels
from experiments.models.cartesian_models import generate_cartesian_prior_from_mes
from experiments.models.polar_models import PolarModels 
from experiments import utils #top level experiments utils
from .utils import generate_perfect_trajectory_np, plot_states_xy #mon-estimation-experiment utils
from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
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
    z0 = np.array([[0.0],   # x0 is row 0 
               [0.0]]) # y0 is row 1
    model = PolarModels.POLAR_COORDINATED_TURN
    #model = CartesianModels.COORDINATED_TURN
    gt_states, detections = generate_perfect_trajectory_np(model, z0, 30)
    plot_states_xy(model, gt_states)
    print(gt_states[:20])
    #M= measure_of_nonlinearity(states, CartesianModels.COORDINATED_TURN.value["trans_mod"], 0.5)
    #print(M)



