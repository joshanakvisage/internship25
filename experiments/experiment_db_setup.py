
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import *
import pandas as pd
from nuscenes.nuscenes import NuScenes

def main():

    #FLOW INDIVIDUAL
    #adjust folder
    #nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes/v1.train-01", verbose=True)
    #1.)INSERT SCENE x=index
    #scene = nusc.scene[x]
    #insert_scene(scene["token"], scene["name"])
    #2.)INSERT INSTANCE y=token of instance
    #insert_instance(nusc, instance["y"])
    #3.)INSERT MOVEMENTS z=movement type ("CONSTANT_VELOCITY" OR "COORDINATED_TURN")
    #insert_movements_from_instance(nusc, instance["token"], "z")


    # Load your Excel file (only the first three columns)
    df = pd.read_excel("/home/internship/Documents/intership25/data/FILTERED_DATA.xlsx", sheet_name="JOSIP", usecols=[0, 1, 3])  
    df.columns = ['scene_index', 'instance_token', 'movement_type']

    nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes/v1.train-01", verbose=True)
    i = 0
    for _, row in df.iterrows():
        if i==0:
            i=i+1
            continue #first row are explanations in xlsx
        scene_index = row['scene_index']
        instance_token = row['instance_token']
        movement_type = row['movement_type']
        #print(f"Scene {int(scene_index)} and instance {instance_token} and {movement_type}")
        scene = nusc.scene[int(scene_index)]  # adjust index if needed
        insert_scene(scene["token"], scene["name"])

        insert_instance(nusc, instance_token)

        # Insert movement (using movement_type from Excel)
        insert_movements_from_instance(nusc, instance_token, movement_type)
    return
  

if __name__ == "__main__":
    #instance = get_instance_by_token("88081a9cb6c74d5190af6d6ea845a6de")
    #print(instance)
    print(len(get_movements_by_instance("88081a9cb6c74d5190af6d6ea845a6de")))
    #main()
