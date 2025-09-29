import numpy as np
import stonesoup
from utils import prepare_movements, kalman
from models.cartesian_models import Models as CartesianModels
from models.polar_models import PolarModels 
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater
from datetime import datetime


class MonData:
    def __init__(self, n, path_length):
        self.paths = [PathData(path_length) for _ in range(n)]
        mon = Mon()
        self.mons = [0 for _ in range(path_length-1)] #NOTE: MON is calculated on one timestamp of the whole dataset
        return
    
    def step(self):
        return

 
class PathData:
    
    def __init__(self, path, n):
        self.start_time = datetime(2025, 9, 12, 12, 0, 0)  #fixed value
        self.GT_PATH = None #Ground truth states from our path
        self.MEAS = None #Sensor measurements from our path (calculated by adding noise to GT)
        self.GT_PATH, self.MEAS = prepare_movements(path)
        self.EST = np.array() 
        self.MSE = np.array()
        self.MET = np.array()

    def calculate_gtpath_and_measpath(self):
        type = "COORDINATED_TURN"
        #type = "POLAR_COORDINATED_TURN"
        if type == "COORDINATED_TURN":
            model_data = CartesianModels.COORDINATED_TURN #TODO: EXTRACT THIS AND HAVE ASME PREDICTOR AND UPDATE FOR POLAR AND CARTESIAN
        elif type == "POLAR_COORDINATED_TURN":
            model_data = PolarModels.POLAR_COORDINATED_TURN

        predictor = UnscentedKalmanPredictor(model_data.value["trans_mod"])
        updater = UnscentedKalmanUpdater(model_data.value["meas_mod"])
            #gt_path, measurements = prepare_movements(movements, model_data)
        self.GT_PATH, self.MES = prepare_movements(movements, model_data, self.start_time, step=0.5, interpolate_points=False)

        
    def calculate_estimated_trajectory(self):
        return

    
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
    movements = None #TODO: FILL WITH STELA'S DATASET
    type = "COORDINATED_TURN"
    