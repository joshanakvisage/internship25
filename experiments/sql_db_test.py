
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.setup_db import create_tables
from database.insert_db import insert_scene, insert_instance, insert_movements_from_instance
from database.query_db import (
    get_all_instances,
    get_movements_by_instance,
    get_instance_by_token, 
    get_scene_by_token
)

# For demo, you’ll need nuscenes-devkit and the dataset installed
from nuscenes.nuscenes import NuScenes

def main():
    # 1. Initialize DB schema
    create_tables()

    # 2. Init NuScenes (replace path with your dataset location)
    nusc = NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=True)

    # 3. Pick a scene
    scene = nusc.scene[0]
    scene_token = scene["token"]
    print(type(scene_token))
    print("type of scene_name")
    print(type(scene["name"]))
    insert_scene(scene_token, scene["name"])

    # 4. Pick an instance from that scene
    instance = nusc.instance[0]   # just grab the first instance for now
    insert_instance(nusc, instance["token"])

    # 5. Insert movements for this instance
    insert_movements_from_instance(nusc, instance["token"])

    # 6. Query back what we inserted
    print("All instances")
    movement = get_movements_by_instance(instance["token"])[0]
    print(movement)
    #print("Movements of this instance:", get_movements_by_instance(instance["token"]))
 

if __name__ == "__main__":
    main()
