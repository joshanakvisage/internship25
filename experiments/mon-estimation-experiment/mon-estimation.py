import numpy as np
import stonesoup
from experiments.models.cartesian_models import Models as CartesianModels
from experiments.models.polar_models import PolarModels
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater
from datetime import datetime
from .utils import generate_perfect_trajectory_np, plot_states_xy, kalman, plot_mons, test_affine_mapping
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
#from experiments import utils #top level experiments utils


class MonData:
    def __init__(self, initial_dataset, path_length, model_type="COORDINATED_TURN"):
        """Initializes the dataset, form a path from each starting data point from the dataset, calculates MoN"""
        self.path_length = path_length
        self.model_type = model_type
        self.paths = [PathData(data_point, self.path_length, self.model_type) for data_point in initial_dataset]
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
        return self.gt_dataset[:, i, :] # shape: (n_paths, state_dim)

    def get_2d_step_est_dataset(self,i):
        """Return estimated state of all paths at step i"""
        if i==0:
            print("No point in calculating for initial dataset as the transition function has not been applied")
        return self.estimated_dataset[:, i, :] # shape: (n_paths, state_dim)

    def calculate_mons(self):
        """Fills array with MoN by dataset layer (timestamp) """
        mon = Mon()

        if self.model_type == "POLAR_COORDINATED_TURN":
            pos_indices = [0, 1]  # x, y in polar
        elif self.model_type == "COORDINATED_TURN":
            pos_indices = [0, 2]  # x, y in Cartesian

        for i in range(1, self.path_length):
            #TODO: WHY ESTIMATED_DATASET - WTF? WHEN GT_PATH IT FAILS
            f = self.gt_dataset[:, i, pos_indices]  # x, y depend on the transition model and prior
            u = self.gt_dataset[:, i-1, pos_indices] # x, y depend on the transition model and prior
            mon_result =  mon.calculate_mon(f, u)
            self.mons.append(float(mon_result))
        return self.mons
    
    def calculate_mon_for_level(self, level_index):
        """
        Calculate MoN for a specific level (step) in the trajectory using only x,y positions.
        level_index: int, the input step (0-based)
        Returns: float, MoN for transition from level_index -> level_index+1
        """
        mon = Mon()

        if self.model_type == "POLAR_COORDINATED_TURN":
            pos_indices = [0, 1]  # x, y in polar
        elif self.model_type == "COORDINATED_TURN":
            pos_indices = [0, 2]  # x, y in Cartesian

        f = self.gt_dataset[:, level_index + 1, pos_indices]  # output step
        u = self.gt_dataset[:, level_index, pos_indices]      # input step

        # Call existing Mon method
        mon_value = mon.calculate_mon(f, u)
        return mon_value

 
class PathData:
    
    def __init__(self, starting_point, path_length, model_type):
        self.start_time = datetime(2025, 9, 12, 12, 0, 0)  #fixed value
        self.starting_point = starting_point #original data set point
        self.path_length = path_length
        self.GT_PATH = None #Ground truth states from our path -> np.ndarray (5,n)
        self.MEAS = [] #Sensor measurements from our path -> array of detection objects
        self.GT_PATH, self.MEAS = self.calculate_gtpath_and_measurements(model_type)
        self.EST = self.calculate_estimated_trajectory()
        self.MSE = None
        self.MET = None

    
    def calculate_gtpath_and_measurements(self, model_type):
        if model_type == "COORDINATED_TURN":
            self.model_data = CartesianModels.COORDINATED_TURN 
        elif model_type == "POLAR_COORDINATED_TURN":
            self.model_data = PolarModels.POLAR_COORDINATED_TURN

        self.predictor = UnscentedKalmanPredictor(self.model_data.value["trans_mod"])
        self.updater = UnscentedKalmanUpdater(self.model_data.value["meas_mod"])
        
        return generate_perfect_trajectory_np(self.model_data, self.starting_point, self.path_length)



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
#T0D0: CHECK THIS IF IT AS 2X1 ARRAY FOR GROUND TRUTH AND DATPATH
def sample_state(alpha=0.3, num_of_points=1000):
    u0 = np.array([1, 10])
    cov_matrix = alpha * np.array([[1, 0], [0, 100]])
    u_rand = np.random.multivariate_normal(mean=u0, cov=cov_matrix, size=num_of_points)
    return [row.reshape(2,1) for row in u_rand]
    







#TODO: EKSTRAKTAT TOČNO INITIAL DATASET ZA PREDIKCIJEU
#TODO: KREĆU IZ ISTE POZICIJE?
if __name__=="__main__":
    # timer_start = datetime.now()
    # #initial_dataset = [np.array([[0.0], [0.0]])]
    initial_dataset = sample_state()
    mondata = MonData(initial_dataset, 30, model_type="COORDINATED_TURN")
    #mondata = MonData(initial_dataset, 30, model_type="COORDINATED_TURN")
    # #print(initial_dataset[:10])
    # #plot_states_xy(mondata.paths[10].GT_PATH)
    # #plot_states_xy(mondata.paths[0].GT_PATH)
    #mons = mondata.calculate_mons()
    #print("ALL MONS")
    #print(mons)
    #plot_mons(mons)
    # #print(mon_2)
    # timer_end = datetime.now() 
    # elapsed = timer_end - timer_start
    # #mon = Mon(level_3_dataset)
    # #result = mon.calculate_mon()
    # print(elapsed)
    # initial_dataset = sample_state(num_of_points=100)
    # #type = "COORDINATED_TURN"
    # type = "POLAR_COORDINATED_TURN"
    # mondata = MonData(initial_dataset, 30, model_type=type)
    # num_to_plot = 1
    # for i in range(num_to_plot):
    #     path = mondata.paths[i]
    #     # The function needs the transition model and states_array
    #     plot_states_xy(path.model_data, path.GT_PATH, color=f"rgba(0,0,255,{0.2*i+0.2})")
    # Take Cartesian CT with w0=0
    #gt = mondata.get_2d_step_gt_dataset(2)  # shape (n_paths, 2)
    #u = mondata.get_2d_step_gt_dataset(1)
    
    # mon_result = mondata.calculate_mon_for_level(15)
    # print(mon_result)

    u_full = mondata.get_2d_step_gt_dataset(2)
    f_full = mondata.get_2d_step_gt_dataset(12)
    res, W, f_hat = test_affine_mapping(u_full, f_full, [0,2])
    print("Affine residual:", res)  