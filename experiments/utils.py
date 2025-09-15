import numpy as np
import matplotlib.pyplot as plt
from stonesoup.plotter import AnimatedPlotterly

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

   


def plot_tracks_with_groundtruth(measurements, groundtruth_path, pred_track, track, tail_length=0.3):
    
    # Extract timestamps
    timestamps = [detection.timestamp for detection in measurements]

    # Create animated plotter
    plotter = AnimatedPlotterly(timestamps, tail_length=tail_length)

    # 
    plotter.plot_ground_truths(
        groundtruth_path,
        mapping=[0, 1],  # X AND Y POSITION INSIDE GROUND TRUTH
        line=dict(color="green")
    )

    # Plot posterior (filtered) track
    plotter.plot_tracks(
        track,
        mapping=[0, 1], # TODO: CHANGE ACCORDING TO PRIOR STRUCTURE
        uncertainty=True
    )

    # Show interactive figure
    plotter.fig.show()