from datetime import timedelta
from experiments.models.cartesian_models import Models as CartesianModels
import numpy as np
from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection


def generate_perfect_trajectory_positions(model_entry, n_steps: int):

    # Assuming 'dt' is a constant internal to this simulation
    dt_seconds = 0.5 
    dt_timedelta = timedelta(seconds=dt_seconds)
    
    trans_model = model_entry.value["trans_mod"]
    current_state = model_entry.value["prior"]
    
    
    positions = np.zeros((n_steps, 2))
    states = []
    
    # Extract initial state and position
    # Assumes state vector is ordered [x, vx, y, vy, ...]
    positions[0, 0] = current_state.state_vector[0, 0]  # x
    positions[0, 1] = current_state.state_vector[2, 0]  # y
    states.append(current_state)

    for i in range(1, n_steps):
        next_state_vector = trans_model.function(
            current_state, noise=False, time_interval=dt_timedelta
        )
        
        # Create the new GaussianState object
        next_state = GaussianState(
            state_vector=next_state_vector,
            covar=current_state.covar.copy(),
            timestamp=current_state.timestamp + dt_timedelta
        )
        
        # Update and store
        current_state = next_state
        states.append(next_state)
        
        # Store position: Assumes state vector is ordered [x, vx, y, vy, ...]
        positions[i, 0] = next_state.state_vector[0, 0]  # x
        positions[i, 1] = next_state.state_vector[2, 0]  # y
        
    return states, positions

def extract_transition_data(states,positions,  transition_model, dt: float):

    dt_timedelta = timedelta(seconds=dt)
    # U = np.hstack([s.state_vector[0,2] for s in states[:-1]]).T
    # F = np.hstack([transition_model.function(s, noise=False, time_interval=dt_timedelta)[0,2]
    #         for s in states[:-1]]).T
    U = positions[:-1].T      # shape: 2 x (n_steps-1)
    F = positions[1:].T       # shape: 2 x (n_steps-1)
    return U, F

def compute_covariances(U, F):
    """
    Computes covariance matrices Σ_uu, Σ_ff, Σ_fu needed for MoN.

    Args:
        U (np.ndarray): Array of current state vectors.
        F (np.ndarray): Array of next state predictions.

    Returns:
        tuple: (Sigma_uu, Sigma_ff, Sigma_fu, U_mean, F_mean)
    """
    U_mean = np.mean(U, axis=1, keepdims=True)
    F_mean = np.mean(F, axis=1, keepdims=True)

    U_centered = U - U_mean
    F_centered = F - F_mean
    
    n_samples = U.shape[1]
    
    # Compute covariance matrices (using 1/(n-1) for sample covariance)
    if n_samples > 1:
        divisor = n_samples - 1
    else:
        # Handle case of only 1 sample (covariance is undefined/zero)
        divisor = 1
        
    Sigma_uu = (U_centered @ U_centered.T) / divisor
    Sigma_ff = (F_centered @ F_centered.T) / divisor
    Sigma_fu = (F_centered @ U_centered.T) / divisor
    
    return Sigma_uu, Sigma_ff, Sigma_fu, U_mean, F_mean

def compute_mon(Sigma_uu, Sigma_ff, Sigma_fu):

    try:
        # Calculate the inverse of Sigma_uu
        inv_Sigma_uu = np.linalg.inv(Sigma_uu)
    except np.linalg.LinAlgError:
        print("Warning: Sigma_uu is singular. Cannot compute MoN.")
        return np.nan

    # Mon numerator (Σ_fu * Σ_uu^-1 * Σ_fu^T) is the linear prediction covariance
    linear_pred_cov = Sigma_fu @ inv_Sigma_uu @ Sigma_fu.T
    
    # M is the trace of the residual covariance (non-linear part)
    M = np.trace(Sigma_ff - linear_pred_cov)
    
    return M

def measure_of_nonlinearity(states, positions, transition_model, dt: float):
    """
    Compute the Measure of Nonlinearity (MoN) for a transition model
    given a list of GaussianState objects.
    """

    U, F = extract_transition_data(states,positions, transition_model, dt)
    
    
    Sigma_uu, Sigma_ff, Sigma_fu, _, _ = compute_covariances(U, F)
    
    
    M = compute_mon(Sigma_uu, Sigma_ff, Sigma_fu)
    
    return M



if __name__ == "__main__":

    model_entry = CartesianModels.COORDINATED_TURN
    transition_model = model_entry.value["trans_mod"]
    
    # --- Trajectory Generation ---
    n_steps = 30
    dt = 0.5
    
    print(f"Generating perfect trajectory for {n_steps} steps...")
    states, positions = generate_perfect_trajectory_positions(model_entry, n_steps)
    
    # Mock plotting
    #plot_positions_numpy(positions)
    


    print("Computing Measure of Nonlinearity (MoN)...")
    M = measure_of_nonlinearity(states, positions, transition_model, dt)
    print(M)
    #print("-" * 30)
    #print(f"Measure of Nonlinearity (M): {M:.4f}")