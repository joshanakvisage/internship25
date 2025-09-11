import sqlite3
from database.setup_db import get_connection

def delete_instance(instance_token: str):
    """Deletes a single instance and all its associated movements. The scene remains intact."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM INSTANCE
        WHERE token = ?
    """, (instance_token,))

    conn.commit()
    conn.close()
    print(f"Instance {instance_token} and its movements deleted.")


def delete_scene(scene_token: str):
    """Deletes a scene and all its associated instances and movements."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM SCENE
        WHERE token = ?
    """, (scene_token,))

    conn.commit()
    conn.close()
    print(f"Scene {scene_token}, all its instances, and their movements deleted.")
