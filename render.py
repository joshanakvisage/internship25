from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes/v1.train-01", verbose=True)

#my_scene = nusc.get('scene', nusc.scene[85]['token'])
my_scene = nusc.get('scene', nusc.scene[51]['token'])

nusc.render_scene_channel(my_scene['token'], 'CAM_FRONT', imsize=(800, 450))




# Get the first sample in the scene
# first_sample_token = my_scene["first_sample_token"]
# first_sample = nusc.get("sample", first_sample_token)

# Render one camera image (instead of whole scene!)
# nusc.render_sample(first_sample["token"], cam_chan="CAM_FRONT")

# Get the first sample in the scene
# first_sample_token = my_scene["first_sample_token"]
# first_sample = nusc.get("sample", first_sample_token)

# Render one camera image (instead of whole scene!)
# nusc.render_sample(first_sample["token"], cam_chan="CAM_FRONT")