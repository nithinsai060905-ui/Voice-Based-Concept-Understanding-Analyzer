import os
import pytest
from database import db_helper

@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Sets up a temporary database file for isolation during tests."""
    test_db_path = "test_analyzer.db"
    
    # Mock the DB_FILE in db_helper to point to our test database
    monkeypatch.setattr(db_helper, "DB_FILE", test_db_path)
    
    # Re-initialize the database schema
    db_helper.init_db()
    
    yield
    
    # Clean up the test database file
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass

def test_save_and_retrieve_setting():
    db_helper.save_setting("test_key", "test_value")
    settings = db_helper.get_settings()
    assert settings.get("test_key") == "test_value"

def test_save_and_retrieve_analysis():
    db_id = db_helper.save_analysis(
        session_name="Test Session",
        audio_filename="test.wav",
        transcript="This is a test transcript.",
        concept_desc="This is the target concept explanation.",
        similarity_score=0.92,
        fluency_score=85.5,
        confidence_score=88.0,
        words_per_minute=120.0,
        filler_count=2,
        pause_count=3,
        duration=15.5,
        concepts_matched=["gravity", "force"],
        concepts_missed=["distance"],
        pdf_report_path="/reports/test.pdf"
    )
    
    assert db_id is not None
    
    history = db_helper.get_analyses_history()
    assert len(history) >= 1
    
    saved_item = history[0]
    assert saved_item["session_name"] == "Test Session"
    assert saved_item["transcript"] == "This is a test transcript."
    assert saved_item["similarity_score"] == 0.92
    assert "gravity" in saved_item["concepts_matched"]
    assert "distance" in saved_item["concepts_missed"]
    assert saved_item["pdf_report_path"] == "/reports/test.pdf"
