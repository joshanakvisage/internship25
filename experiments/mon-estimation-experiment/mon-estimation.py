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
    def __init__(self, data_set_size, path_length):
        """Initializes the dataset, form a path from each starting data point from the dataset, calculates MoN"""
        dataset = [[0.0, 0.0]]
        path_length
        self.paths = [PathData(data_point, path_length) for data_point in dataset]
        self.mons = [Mon() for _ in range(path_length-1)] #NOTE: MON is calculated on one timestamp of the whole dataset
        return
    
    def step():
        """Should collect gt_point from all paths for a specific step in the trajectory"""
        return

 
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
    # type = "COORDINATED_TURN"
    timer_start = datetime.now()
    z0 = np.array([[0.0],   # x0 is row 0 
               [0.0]]) # y0 is row 1
    path = PathData(z0, 40)
    timer_end = datetime.now() 
    print(path.GT_PATH[:20])
    elapsed = timer_end - timer_start
    print(f"Time to initialize 1 path: {elapsed.total_seconds():.6f} seconds")
    #plot_states_xy(path.GT_PATH)