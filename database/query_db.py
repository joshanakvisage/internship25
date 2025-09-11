import sqlite3
from database.setup_db import get_connection

def get_all_instances():
    """Return all instances in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT token, category FROM INSTANCE")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_scene_by_token(scene_token):
    """Return scene connected to token."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT token, name FROM SCENE WHERE token = ?", (str(scene_token),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_instance_by_token(instance_token):
    """Return instance connected to token"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT token, category, scene_token
        FROM INSTANCE
        WHERE token = ?
    """, (instance_token,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_movements_by_instance(instance_token):
    """Return all movements/annotations connected to instance token"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT annotation_token, movement_type, translation, rotation, timestamp, velocity
        FROM MOVEMENT
        WHERE instance_token = ?
        ORDER BY timestamp
    """, (instance_token,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_all_scenes():
    """Return all scenesin the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT token, name FROM SCENE")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]