from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes')

my_scene = nusc.get('scene', nusc.scene[2]['token'])

nusc.render_scene_channel(my_scene['token'], 'CAM_FRONT', imsize=(1600, 900))