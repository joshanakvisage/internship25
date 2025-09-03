import sqlite3

DB_PATH = "nuscenes_filtered.db"

def get_connection():
    #Returns a SQLite connection with foreign keys enabled.
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    # SCENE table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS SCENE (
            token INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        ) 
    """)

    # INSTANCE table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS INSTANCE (
            token INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            scene_token INTEGER NOT NULL,
            FOREIGN KEY (scene_token) REFERENCES SCENE(token) ON DELETE CASCADE
        )
    """)

    # MOVEMENT table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS MOVEMENT (
            annotation_token INTEGER PRIMARY KEY,
            movement_type TEXT NOT NULL,
            translation TEXT NOT NULL,
            rotation TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            velocity REAL NOT NULL,
            instance_token INTEGER NOT NULL,
            FOREIGN KEY (instance_token) REFERENCES INSTANCE(token) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_tables()