import numpy as np
import stonesoup
from experiments.models.cartesian_models import Models as CartesianModels
from experiments.models.polar_models import PolarModels
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater
from datetime import datetime
from .utils import generate_perfect_trajectory_positions, generate_perfect_trajectory_np, plot_states_xy, kalman
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
#from experiments import utils #top level experiments utils


class MonData:
    def __init__(self, initial_dataset, path_length, type="COORDINATED_TURN"):
        """Initializes the dataset, form a path from each starting data point from the dataset, calculates MoN"""
        self.path_length = path_length
        self.paths = [PathData(data_point, self.path_length, type) for data_point in initial_dataset]
        self.gt_dataset, self.estimated_dataset = None, None
        self.make_gt_and_estimated_dataset()
        self.mons = []  #NOTE: MON is calculated on one timestamp of the whole dataset
        return
    
    def make_gt_and_estimated_dataset(self):
        """gt_dataset is 3d matrix -> stacked 2d matrix of all paths and its state vectors, indexed by timestamp -> the same for self.estimated_dataset"""
        self.gt_dataset = np.stack([path.GT_PATH for path in self.paths], axis=0)  
        self.estimated_dataset = np.stack([path.EST for path in self.paths], axis=0)
   
    #if
    # n_paths = 100
    # path_length = 40
    # state_dim = 5
    #then
    # self.gt_dataset.shape == (100, 40, 5)
    # self.get_2d_step_gt_dataset(10).shape == (100, 5) -> all ground truth paths, step 10
    # self.get_2d_step_est_dataset(10).shape == (100, 5) -> all trajectories, step 10


    def get_2d_step_gt_dataset(self,i):
        """Return ground-truth state of all paths at step i"""
        if i==0:
            return self.initial_dataset
        return self.gt_dataset[:, i, :] # shape: (n_paths, state_dim)

    def get_2d_step_est_dataset(self,i):
        """Return estimated state of all paths at step i"""
        if i==0:
            print("No point in calculating for initial dataset as the transition function has not been applied")
        return self.estimated_dataset[:, i, :] # shape: (n_paths, state_dim)

    def calculate_mons(self):
        """Fills array with MoN by dataset layer (timestamp) """
        mon = Mon()
        for i in range(1, self.path_length):
            f = self.estimated_dataset[:, i, [0, 2]]  # x = 0, y = 2
            u = self.estimated_dataset[:, i-1, [0, 2]]
            mon_result =  mon.calculate_mon(f, u)
            self.mons.append(mon_result)
        return self.mons

 
class PathData:
    
    def __init__(self, starting_point, path_length, type):
        self.start_time = datetime(2025, 9, 12, 12, 0, 0)  #fixed value
        self.starting_point = starting_point #original data set point
        self.path_length = path_length
        self.GT_PATH = None #Ground truth states from our path -> np.ndarray (5,n)
        self.MEAS = [] #Sensor measurements from our path -> array of detection objects
        self.GT_PATH, self.MEAS = self.calculate_gtpath_and_measurements(type)
        self.EST = self.calculate_estimated_trajectory()
        self.MSE = None
        self.MET = None

    
    def calculate_gtpath_and_measurements(self, type):
        if type == "COORDINATED_TURN":
            self.model_data = CartesianModels.COORDINATED_TURN 
        elif type == "POLAR_COORDINATED_TURN":
            self.model_data = PolarModels.POLAR_COORDINATED_TURN

        self.predictor = UnscentedKalmanPredictor(self.model_data.value["trans_mod"])
        self.updater = UnscentedKalmanUpdater(self.model_data.value["meas_mod"])
        
        return generate_perfect_trajectory_np(CartesianModels.COORDINATED_TURN, self.starting_point, self.path_length)



    def calculate_estimated_trajectory(self):
        apriori_track, aposteriori_track, posterior_array = kalman(self.MEAS, self.model_data, self.predictor, self.updater)
        return posterior_array

    
    def get_MSE(self): #should i jut calculate this out of numpy 
        return
    


class Mon():
    def __init__(self):
        pass


    def autocovariance(self, X):
        X_centered = X - np.mean(X, axis=0)
        return np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)

    def cross_covariance(self,X, Y):
        X_centered = X - np.mean(X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)
        return np.dot(X_centered.T, Y_centered) / (X.shape[0] - 1)

    def get_function_output(self):
        return
    
    def calculate_mon(self, f,u):
        cov_ff = self.autocovariance(f)
        cov_fu = self.cross_covariance(f, u)
        cov_uf = cov_fu.T
        cov_uu = self.autocovariance(u)
        residual_cov = cov_ff - cov_fu @ np.linalg.inv(cov_uu) @ cov_uf
        mon = np.sqrt(np.trace(residual_cov)) / np.sqrt(np.trace(cov_ff))
        return mon

def sample_state(alpha=0.2, num_of_points=1000):
    u0 = np.array([1, 10])
    cov_matrix = alpha * np.array([[1, 0], [0, 100]])
    u_rand = np.random.multivariate_normal(mean=u0, cov=cov_matrix, size=num_of_points)
    return [row.reshape(2,1) for row in u_rand]
    

if __name__=="__main__":
    timer_start = datetime.now()
    #initial_dataset = [np.array([[0.0], [0.0]])]
    initial_dataset = sample_state()
    mondata = MonData(initial_dataset, 30)
    mon_2 = mondata.calculate_mons()[1]
    print(mon_2)
    timer_end = datetime.now() 
    elapsed = timer_end - timer_start
    #mon = Mon(level_3_dataset)
    #result = mon.calculate_mon()
    print(elapsed)
   

