
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import *

from nuscenes.nuscenes import NuScenes

def main():
    return
    #FLOW
    #adjust folder
    #nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes/v1.train-01", verbose=True)
    #1.)INSERT SCENE x=index
    #scene = nusc.scene[x]
    #insert_scene(scene["token"], scene["name"])
    #2.)INSERT INSTANCE y=token of instance
    #insert_instance(nusc, instance["y"])
    #3.)INSERT MOVEMENTS z=movement type ("CONSTANT_VELOCITY" OR "COORDINATED_TURN")
    #insert_movements_from_instance(nusc, instance["token"], "z")

  

if __name__ == "__main__":
    main()
