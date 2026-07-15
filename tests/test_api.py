import os
import pytest
from fastapi.testclient import TestClient
from database import db_helper

@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Isolates testing to a temporary database file."""
    test_db_path = "test_api_analyzer.db"
    monkeypatch.setattr(db_helper, "DB_FILE", test_db_path)
    db_helper.init_db()
    
    yield
    
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass

def test_api_auth_flow():
    # Import inside test to ensure DB_FILE patch is applied first
    from backend.main import app
    client = TestClient(app)
    
    # 1. Register User
    reg_response = client.post("/register", json={"username": "testuser", "password": "testpassword"})
    assert reg_response.status_code == 201
    assert reg_response.json()["username"] == "testuser"
    
    # 2. Login User
    login_response = client.post("/login", data={"username": "testuser", "password": "testpassword"})
    assert login_response.status_code == 200
    token_data = login_response.json()
    assert "access_token" in token_data
    token = token_data["access_token"]
    
    # 3. Retrieve history (should be empty initially)
    headers = {"Authorization": f"Bearer {token}"}
    history_response = client.get("/history", headers=headers)
    assert history_response.status_code == 200
    assert len(history_response.json()) == 0
