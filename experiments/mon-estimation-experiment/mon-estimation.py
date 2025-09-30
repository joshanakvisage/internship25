import numpy as np
import stonesoup
from experiments.models.cartesian_models import Models as CartesianModels
from experiments.models.polar_models import PolarModels
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater
from datetime import datetime
from .utils import generate_perfect_trajectory_positions, generate_perfect_trajectory_np, plot_states_xy, kalman
#from experiments import utils #top level experiments utils


class MonData:
    def __init__(self, initial_dataset, path_length):
        """Initializes the dataset, form a path from each starting data point from the dataset, calculates MoN"""
        self.paths = [PathData(data_point, path_length) for data_point in initial_dataset]
        self.make_gt_and_estimated_dataset()
        #self.mons = [Mon() for _ in range(path_length-1)] #NOTE: MON is calculated on one timestamp of the whole dataset
        return
    
    def make_gt_and_estimated_dataset(self):
        # shape: (n_paths, n_steps, state_dim)
        self.gt_dataset = np.stack([path.GT_PATH for path in self.paths], axis=0)  
        self.estimated_dataset = np.stack([path.EST for path in self.paths], axis=0)
   
    #if
    # n_paths = 100
    # path_length = 40
    # state_dim = 5
    #then
    # self.gt_dataset.shape == (100, 40, 5)
    # self.get_2d_step_gt_dataset(10).shape == (100, 5) -> all paths, step 10
    # self.get_2d_step_est_dataset(10).shape == (100, 5) -> all paths, step 10


    def get_2d_step_gt_dataset(self,i):
        """Return ground-truth state of all paths at step i"""
        if i==0:
            return self.initial_dataset
        return self.gt_dataset[:, i, :] 

    def get_2d_step_est_dataset(self,i):
        """Return estimated state of all paths at step i"""
        if i==0:
            print("No point in calculating for initial dataset as the transition function has not been applied")
        return self.estimated_dataset[:, i, :] # shape: (n_paths, state_dim)

    def get_data_step(self, i):
        """Retrieves ground truth data set for all points for path step and estimated dataset for path step """
        if i==0:
            print("First position the same for estimated and groundtruth")
        for path in self.paths:
            self.gt_dataset.append(path.GT_PATH[i])
            self.estimated_dataset.append(path.EST[i])

 
class PathData:
    
    def __init__(self, starting_point, path_legth):
        self.start_time = datetime(2025, 9, 12, 12, 0, 0)  #fixed value
        self.starting_point = starting_point #original data set point
        self.path_length = path_legth
        self.GT_PATH = None #Ground truth states from our path -> np.ndarray (5,n)
        self.MEAS = [] #Sensor measurements from our path -> array of detection objects
        self.GT_PATH, self.MEAS = self.calculate_gtpath_and_measurements()
        self.EST = self.calculate_estimated_trajectory()
        self.MSE = None
        self.MET = None

    def calculate_gtpath_and_measurements(self):
        type = "COORDINATED_TURN"
        #type = "POLAR_COORDINATED_TURN"
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
    def __init__(self, dataset):
        self.dataset = dataset
        self.result = None
        return 0

    def autocovariance(self):
        X_centered = self.dataset - np.mean(self.dataset, axis=0)
        return np.dot(X_centered.T, X_centered) / (self.dataset.shape[0] - 1)

    def cross_covariance(X, Y):
        X_centered = X - np.mean(X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)
        return np.dot(X_centered.T, Y_centered) / (X.shape[0] - 1)

    def calculate_mon(self, f,u):
        cov_ff = self.autocovariance(f)
        cov_fu = self.cross_covariance(f, u)
        cov_uf = cov_fu.T
        cov_uu = self.autocovariance(u)
        residual_cov = cov_ff - cov_fu @ np.linalg.inv(cov_uu) @ cov_uf
        mon = np.sqrt(np.trace(residual_cov)) / np.sqrt(np.trace(cov_ff))
        return mon
    

if __name__=="__main__":
    # movements = None #TODO: FILL WITH STELA'S DATASET
    type = "COORDINATED_TURN"
    #timer_start = datetime.now()
    z0 = np.array([[0.0],   # x0 is row 0 
                [0.0]]) # y0 is row 1
    path = PathData(z0, 30)
    # timer_end = datetime.now() 
    print(path.EST[4])
    # elapsed = timer_end - timer_start
    # print(f"Time to initialize 1 path: {elapsed.total_seconds():.6f} seconds")
    #plot_states_xy(path.GT_PATH)
    timer_start = datetime.now()
    initial_dataset = [np.array([[0.0], [0.0]])]
    mon = MonData(initial_dataset, 30)
    timer_end = datetime.now() 
    elapsed = timer_end - timer_start
    print(mon.get_2d_step_est_dataset(4))
