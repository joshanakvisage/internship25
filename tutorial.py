import numpy as np
from datetime import datetime, timedelta


from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
import plotly.io as pio
pio.renderers.default = "browser"
# And the clock starts
start_time = datetime.now().replace(microsecond=0)


np.random.seed(1991)


q_x = 0.05
q_y = 0.05
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])



timesteps = [start_time]
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])

num_steps = 20
for k in range(1, num_steps + 1):

    timesteps.append(start_time+timedelta(seconds=k))  # add next timestep to list of timesteps
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))



from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
plotter.plot_ground_truths(truth, [0, 2])
#plotter.fig.show()


transition_model.matrix(time_interval=timedelta(seconds=1))

transition_model.covar(time_interval=timedelta(seconds=1))

print(transition_model)



from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian


measurement_model = LinearGaussian(
    ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0, 2),  # Mapping measurement vector index to state index
    noise_covar=np.array([[5.0,0], # Covariance matrix for Gaussian PDF
                          [0, 1e-4]])
    )


measurement_model.matrix()

measurement_model.covar()


measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement,
                                  timestamp=state.timestamp,
                                  measurement_model=measurement_model))



plotter.plot_measurements(measurements, [0, 2])
#plotter.fig.show()


#KALMAN FILTER PART
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)


from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

from stonesoup.types.hypothesis import SingleHypothesis


from stonesoup.types.track import Track
track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]



""" plotter.plot_tracks(track, [0, 2], uncertainty=True)
plotter.fig.show()  """

from stonesoup.types.track import Track
pred_track = Track()

prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    pred_track.append(prediction)  # store the prediction
    
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)
    print(f"Measurement {measurement.state_vector} prediction {prediction.state_vector}")
    prior = post

plotter.plot_tracks(pred_track, [0, 2], uncertainty=True, line=dict(color="orange"))
plotter.plot_tracks(track, [0, 2], uncertainty=True)
plotter.fig.show()