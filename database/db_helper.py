import sqlite3
import json
import os

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzer.db")

def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Analyses Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        session_name TEXT NOT NULL,
        audio_filename TEXT,
        transcript TEXT,
        concept_desc TEXT,
        similarity_score REAL,
        fluency_score REAL,
        confidence_score REAL,
        words_per_minute REAL,
        filler_count INTEGER,
        pause_count INTEGER,
        duration REAL,
        concepts_matched TEXT,  -- JSON list of strings
        concepts_missed TEXT,   -- JSON list of strings
        pdf_report_path TEXT
    )
    """)
    
    # Settings Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)
    
    # Seed default settings
    default_settings = [
        ("whisper_model", "tiny"),
        ("similarity_model", "all-MiniLM-L6-v2"),
        ("similarity_threshold", "0.60"),
        ("filler_words", "um,uh,like,you know,so,actually,basically,mean,well"),
        ("silence_threshold_db", "-35")
    ]
    for key, value in default_settings:
        cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))
        
    conn.commit()
    conn.close()

def save_analysis(session_name, audio_filename, transcript, concept_desc, 
                  similarity_score, fluency_score, confidence_score, 
                  words_per_minute, filler_count, pause_count, duration, 
                  concepts_matched, concepts_missed, pdf_report_path):
    """Saves a new concept understanding analysis record to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO analyses (
        session_name, audio_filename, transcript, concept_desc, 
        similarity_score, fluency_score, confidence_score, 
        words_per_minute, filler_count, pause_count, duration, 
        concepts_matched, concepts_missed, pdf_report_path
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_name, audio_filename, transcript, concept_desc,
        similarity_score, fluency_score, confidence_score,
        words_per_minute, filler_count, pause_count, duration,
        json.dumps(concepts_matched), json.dumps(concepts_missed),
        pdf_report_path
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_analyses_history():
    """Retrieves all past analyses ordered by timestamp descending."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    
    history = []
    for r in rows:
        item = dict(r)
        item["concepts_matched"] = json.loads(item["concepts_matched"]) if item["concepts_matched"] else []
        item["concepts_missed"] = json.loads(item["concepts_missed"]) if item["concepts_missed"] else []
        history.append(item)
        
    conn.close()
    return history

def delete_analysis(analysis_id):
    """Deletes an analysis record from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()

def get_settings():
    """Fetches all configurations as a dictionary."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM settings")
    rows = cursor.fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}

def save_setting(key, value):
    """Saves or updates a configuration setting."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()

# Initialize on import
init_db()
