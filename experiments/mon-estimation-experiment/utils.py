from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
import numpy as np
from experiments.models.cartesian_models import Models as CartesianModels
from experiments.models.cartesian_models import generate_cartesian_prior_from_mes
from experiments.models.polar_models import PolarModels 
from experiments import utils #top level experiments utils
from datetime import timedelta


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
    
   
    #TODO: take this and set it up for self.gt_path and self.mes inside the mon-estimation
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

