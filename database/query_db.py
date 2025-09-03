import sqlite3
from database.setup_db import get_connection


def get_scene_by_token(scene_token):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT token, name FROM SCENE WHERE token = ?", (scene_token,))
    result = cursor.fetchone()
    conn.close()
    return result


def get_instance_by_token(instance_token):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT token, category, scene_token
        FROM INSTANCE
        WHERE token = ?
    """, (instance_token,))
    result = cursor.fetchone()
    conn.close()
    return result

def get_movements_by_instance(instance_token):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT annotation_token, movement_type, translation, rotation, timestamp, velocity
        FROM MOVEMENT
        WHERE instance_token = ?
        ORDER BY timestamp
    """, (instance_token,))
    results = cursor.fetchall()
    conn.close()
    return results

#TODO: TEST EVERYTHING