import numpy as np
from nuscenes.nuscenes import NuScenes

def get_xy_simple(nusc, scene_index, instance_token):
    scene = nusc.scene[scene_index]
    sample_token = scene['first_sample_token']

    xs, ys = [], []

    while sample_token:
        sample = nusc.get('sample', sample_token)
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            if ann['instance_token'] == instance_token:
                x, y, _ = ann['translation']
                xs.append(x)
                ys.append(y)
                break
        sample_token = sample['next'] if sample['next'] else None

    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-trainval', dataroot='data', verbose=False)

    # List of (scene_index, instance_token) for 5 scenes
    scenes_tokens = [
        (788, "3c706e750241410e9acdc1244608db77"),
        (738, "81a1e046467e41afb31614485f837b44"),
        (739, "d73eabe9c98345d4aa917704926e5764"),
        (673, "3af35a4f9a0846c99aacf43ca02b773f"),
        (823, "e6cdefc1f8cb42deae24c7f11c208415"),
    ]  

    data = {}
    for scene_idx, token in scenes_tokens:
        x, y = get_xy_simple(nusc, scene_idx, token)
        data[(scene_idx, token)] = (x, y)
        print(f"Scene {scene_idx}, Token {token}: {len(x)} points extracted")

    # Now `data` holds (x,y) arrays for each scene and token
all_x = np.concatenate([xy[0] for xy in data.values()])
all_y = np.concatenate([xy[1] for xy in data.values()])

print(f"Total points in x: {len(all_x)}")
print(f"Total points in y: {len(all_y)}")

mean_x = np.mean(all_x)
mean_y = np.mean(all_y)

print(f"Mean of all x positions: {mean_x}")
print(f"Mean of all y positions: {mean_y}")

mean_vector = np.array([np.mean(all_x), np.mean(all_y)])
cov_matrix = np.cov(all_x, all_y)

print(f"Mean vector: {mean_vector}")
print(f"Covariance matrix:\n{cov_matrix}")
-
def sample_state_with_data(alpha, mean_vec, cov_mat, num_of_points=10000):
    u0 = mean_vec
    cov_matrix = alpha * cov_mat
    u_rand = np.random.multivariate_normal(mean=u0, cov=cov_matrix, size=num_of_points)
    return u_rand

samples = sample_state_with_data(alpha=1, mean_vec=mean_vector, cov_mat=cov_matrix)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(samples[:,0], samples[:,1], s=1, alpha=0.3)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of sampled 2D points')
plt.grid(True)
plt.show()