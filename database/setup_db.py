import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "nuscenes_filtered.db")


def get_connection():
    #Returns a SQLite connection with foreign keys enabled.
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row   #TODO: CHECK IF THIS IS CORRECT
    return conn


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    # SCENE table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS SCENE (
            token TEXT PRIMARY KEY,
            name TEXT NOT NULL
        ) 
    """)

    # INSTANCE table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS INSTANCE (
            token TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            scene_token TEXT NOT NULL,
            FOREIGN KEY (scene_token) REFERENCES SCENE(token) ON DELETE CASCADE
        )
    """)

    # MOVEMENT table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS MOVEMENT (
            annotation_token TEXT PRIMARY KEY,
            movement_type TEXT NOT NULL,
            translation TEXT NOT NULL,
            rotation TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            velocity TEXT NOT NULL,
            instance_token TEXT NOT NULL,
            FOREIGN KEY (instance_token) REFERENCES INSTANCE(token) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_tables()