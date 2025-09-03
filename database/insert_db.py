from database import get_connection 

def insert_scene(scene_token, name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO SCENE (token, name) VALUES (?, ?)",
                   (scene_token, name))
    conn.commit()
    conn.close()

def insert_instance(instance_token, nusc):
    conn = get_connection()
    cursor = conn.cursor()
    
    instance = nusc.get('instance', instance_token)
    ann_object = nusc.get('sample_annotation', instance['first_annotation_token'])

    category = ann_object['category_name']
    scene_token = nusc.get('sample', ann_object['sample_token'])['scene_token']

    cursor.execute("INSERT OR IGNORE INTO INSTANCE (token, category, scene_token) VALUES (?, ?, ?)",
                   (instance_token, category, scene_token))
    conn.commit()
    conn.close()

def insert_movements_from_instance(nusc, instance_token):
    instance = nusc.get('instance', instance_token)
    first_ann_token = instance['first_annotation_token']
    last_ann_token = instance['last_annotation_token']

    conn = get_connection()
    cursor = conn.cursor()

    current_ann_token = first_ann_token
    while True:
        ann_object = nusc.get('sample_annotation', current_ann_token)
        cursor.execute("""
            INSERT OR IGNORE INTO MOVEMENT
            (annotation_token, movement_type, translation, rotation, timestamp, velocity, instance_token)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ann_object['token'],
            str(nusc.get('attribute', ann_object['attribute_tokens'][0])['name']), #TODO:ATTRIBUTE OR SHOULD I CALCULATE IT  ->F 
            str(ann_object['translation']),                  #TODO: CHECK STORE AS STRING OR?
            str(ann_object['rotation']),
            ann_object['timestamp'],                         #TODO: CHECK CONVERSION!
            nusc.box_velocity(ann_object['token']),
            ann_object['instance_token']   #NOTE: FOREIGN KEY FOR INSTANCE
        ))

        if current_ann_token == last_ann_token:
            break
        current_ann_token = ann_object["next"]

    conn.commit()
    conn.close()



