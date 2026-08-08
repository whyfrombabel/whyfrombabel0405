import sqlite3
import os

DB_PATH = 'melomind.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            query_type TEXT NOT NULL,
            query_data TEXT,
            results_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

def query_history(user_id):
    conn = get_db()
    rows = conn.execute(
        'SELECT id, query_type, query_data, created_at FROM search_history WHERE user_id=? ORDER BY created_at DESC LIMIT 20',
        (user_id,)
    ).fetchall()
    conn.close()
    return rows

def save_history(user_id, qtype, qdata, results):
    conn = get_db()
    conn.execute(
        'INSERT INTO search_history (user_id, query_type, query_data, results_json) VALUES (?,?,?,?)',
        (user_id, qtype, qdata, results)
    )
    conn.commit()
    conn.close()
