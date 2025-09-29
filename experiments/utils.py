import numpy as np
import matplotlib.pyplot as plt
from stonesoup.plotter import AnimatedPlotterly
from datetime import timedelta, datetime
import numpy as np
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from database import *

#Yaw only important for vehicles in 2d (rotation around the vertical axis)
def quaternion_to_yaw(q):
    w, x, y, z = q
    # yaw (z-axis rotation)
    yaw = np.arctan2(2.0 * (w*z + x*y),
                     1.0 - 2.0 * (y*y + z*z))
    return yaw


def extract_state_data(gt_path, track, type):
    #Extract ground truth and aposteriori data to better format
    
    # Ground truth which is pulled from the database (was inserted as x,y,yaw,vx,vz)
    gt_x = [state.state_vector[0, 0] for state in gt_path]
    gt_y = [state.state_vector[1, 0] for state in gt_path]
    gt_vx = [state.state_vector[3, 0] for state in gt_path]
    gt_vy = [state.state_vector[4, 0] for state in gt_path]
    times = [state.timestamp for state in gt_path]

    # Posterior estimates
    if type in ["CONSTANT_VELOCITY", "COORDINATED_TURN"]:
        print("HELLO")
        est_x = [state.state_vector[0, 0] for state in track]
        est_y = [state.state_vector[2, 0] for state in track]
        est_vx = [state.state_vector[1, 0] for state in track]
        est_vy = [state.state_vector[3, 0] for state in track]

    elif type == "POLAR_COORDINATED_TURN":
        est_x = [state.state_vector[0, 0] for state in track]
        est_y = [state.state_vector[1, 0] for state in track]
        est_v = [state.state_vector[2, 0] for state in track]
        est_h = [state.state_vector[3, 0] for state in track]
        est_vx = est_v*np.cos(est_h)
        est_vy = est_v*np.sin(est_h)

    return {
        "gt_x": gt_x,
        "gt_y": gt_y,
        "gt_vx": gt_vx,
        "gt_vy": gt_vy,
        "est_x": est_x,
        "est_y": est_y,
        "est_vx": est_vx,
        "est_vy": est_vy,
        "times": times,
    }


def plot_velocity(data, axis="x"):
    #Plot ground truth vs posterior velocity for X or Y axis.
    times = data["times"]
    gt_v = data[f"gt_v{axis.lower()}"]
    est_v = data[f"est_v{axis.lower()}"]

    plt.figure(figsize=(10, 5))
    plt.plot(times, gt_v, label=f"Ground Truth V{axis.upper()}", color="blue")
    plt.plot(times, est_v, label=f"Posterior V{axis.upper()}", color="red", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel(f"Velocity {axis.upper()}")
    plt.title(f"Ground Truth vs Posterior {axis.upper()} Velocity")
    plt.legend()
    plt.grid(True)


def plot_position(data, axis="x"):
    #Plot ground truth vs posterior position for X or Y axis.
    times = data["times"]
    gt_pos = data[f"gt_{axis.lower()}"]
    est_pos = data[f"est_{axis.lower()}"]

    plt.figure(figsize=(10, 5))
    plt.plot(times, gt_pos, label=f"Ground Truth {axis.upper()}", color="blue")
    plt.plot(times, est_pos, label=f"Posterior {axis.upper()}", color="red", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel(f"{axis.upper()} Position")
    plt.title(f"Ground Truth vs Posterior {axis.upper()}")
    plt.legend()
    plt.grid(True)


def plot_combined_tracks(data):
    """
    Plot ground truth and posterior tracks in one figure with 4 subplots:
    1. Velocity X
    2. Velocity Y
    3. Position X
    4. Position Y
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_vel_x, ax_vel_y, ax_pos_x, ax_pos_y = axes.flatten()
    
    times = data["times"]

    # Velocity X
    ax_vel_x.plot(times, data["gt_vx"], label="GT Velocity X", color="green")
    ax_vel_x.plot(times, data["est_vx"], label="Posterior Velocity X", color="blue")
    ax_vel_x.set_title("Velocity X")
    ax_vel_x.set_xlabel("Time [s]")
    ax_vel_x.set_ylabel("Velocity [units/s]")
    ax_vel_x.legend()
    ax_vel_x.grid(True)

    # Velocity Y
    ax_vel_y.plot(times, data["gt_vy"], label="GT Velocity Y", color="green")
    ax_vel_y.plot(times, data["est_vy"], label="Posterior Velocity Y", color="blue")
    ax_vel_y.set_title("Velocity Y")
    ax_vel_y.set_xlabel("Time [s]")
    ax_vel_y.set_ylabel("Velocity [units/s]")
    ax_vel_y.legend()
    ax_vel_y.grid(True)

    # Position X
    ax_pos_x.plot(times, data["gt_x"], label="GT Position X", color="green")
    ax_pos_x.plot(times, data["est_x"], label="Posterior Position X", color="blue")
    ax_pos_x.set_title("Position X")
    ax_pos_x.set_xlabel("Time [s]")
    ax_pos_x.set_ylabel("Position [units]")
    ax_pos_x.legend()
    ax_pos_x.grid(True)

    # Position Y
    ax_pos_y.plot(times, data["gt_y"], label="GT Position Y", color="green")
    ax_pos_y.plot(times, data["est_y"], label="Posterior Position Y", color="blue")
    ax_pos_y.set_title("Position Y")
    ax_pos_y.set_xlabel("Time [s]")
    ax_pos_y.set_ylabel("Position [units]")
    ax_pos_y.legend()
    ax_pos_y.grid(True)

    plt.tight_layout()
    plt.show()
   

#NOTE: CHANGE PLOTTER. PLOT TRACKS ACCORDING TO APOSTERIOR STATE VECTOR
def plot_tracks_with_groundtruth(measurements, groundtruth_path, pred_track, track, tail_length=0.3):

    timestamps = [detection.timestamp for detection in measurements]

    # Create animated plotter
    plotter = AnimatedPlotterly(timestamps, tail_length=tail_length)

    plotter.plot_ground_truths(
        groundtruth_path,
        mapping=[0, 1],  # X AND Y POSITION INSIDE GROUND TRUTH
        line=dict(color="red")
    )

    # Plot posterior (filtered) track
    plotter.plot_tracks(
        track,
        mapping=[0, 2], # NOTE: CHANGE ACCORDING TO STATE VECTOR STRUCTURE -> VISIBLE IN MODELS
        uncertainty=True
    )

    # Show interactive figure
    plotter.fig.show()


#check if indexes are correct
def prepare_movements(movements, model_data, start_time, step=0.5, interpolate_points=False):
    """
    Convert movement dictionaries into StoneSoup GroundTruthPath & Detection list.
    Optionally interpolates an extra midpoint measurement between each movement.
    """
    detections, gt_states = [], []
    R = model_data.value["meas_mod"].noise_covar

    for i, mov in enumerate(movements):
        # parse strings to arrays once
        parsed_mov = {
            "translation": str_to_array(mov["translation"]),
            "velocity": str_to_array(mov["velocity"]),
            "rotation": str_to_array(mov["rotation"])
        }

        t = start_time + timedelta(seconds=i * step)
        gt_state, detection = make_gt_and_det(parsed_mov, t, R)
        gt_states.append(gt_state)
        detections.append(detection)

        if interpolate_points and i < len(movements) - 1:
            next_mov = movements[i + 1]
            # parse next once
            next_parsed = {
                "translation": str_to_array(next_mov["translation"]),
                "velocity": str_to_array(next_mov["velocity"]),
                "rotation": str_to_array(next_mov["rotation"])
            }
            mid_mov = make_midpoint_mov(parsed_mov, next_parsed)
            t_mid = start_time + timedelta(seconds=(i + 0.5) * step)
            gt_state_mid, detection_mid = make_gt_and_det(mid_mov, t_mid, R)
            gt_states.append(gt_state_mid)
            detections.append(detection_mid)

    gt_path = GroundTruthPath(gt_states)
    return gt_path, detections


def make_gt_and_det(mov, timestamp, R):
    """mov is expected to already contain numpy arrays (not strings)."""
    pos_2d = mov["translation"][:2].reshape(-1, 1)
    vel_2d = mov["velocity"][:2].reshape(-1, 1)
    yaw = quaternion_to_yaw(mov["rotation"])  # mov["rotation"] already np.array

    gt_state = GroundTruthState(
        state_vector=np.concatenate([pos_2d, np.array([[yaw]]), vel_2d], axis=0),
        timestamp=timestamp
    )
    noisy_state = make_noisy(pos_2d, R) if R is not None else pos_2d
    detection = Detection(state_vector=noisy_state, timestamp=timestamp)
    return gt_state, detection


def make_midpoint_mov(mov1, mov2):
    """
    Return a new movement dict (numpy arrays) halfway between mov1 and mov2.
    """
    mid_trans = (mov1["translation"] + mov2["translation"]) / 2.0
    mid_vel = (mov1["velocity"] + mov2["velocity"]) / 2.0
    mid_rot = (mov1["rotation"] + mov2["rotation"]) / 2.0  # still quaternion array; replace with yaw avg if needed

    return {
        "translation": mid_trans,
        "velocity": mid_vel,
        "rotation": mid_rot
    }


def make_noisy(data, R): #for now takes in the measurement model matrix - is this correct?
    mean = np.zeros(R.shape[0])
    noise = np.random.multivariate_normal(mean, R).reshape(-1, 1)
    return data+noise


def str_to_array(s):
    #Converts array in a string type (like movement,translation etc) to np.array for stonesoup
    if not s:
        return np.array([])
    return np.fromstring(s.strip("[]"), sep=',')