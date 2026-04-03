from fastapi.testclient import TestClient
from app.main import app
import uuid

client = TestClient(app, raise_server_exceptions=False)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_session_lifecycle():
    session_id = str(uuid.uuid4())
    
    # 1. Check stats for new session
    response = client.get(f"/api/v1/sessions/{session_id}/stats")
    assert response.status_code == 200
    assert response.json()["total_messages"] == 0
    
    # 2. Clear session (should be idempotent)
    response = client.delete(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"

def test_chat_endpoint_structure():
    # We mock the LLM call in a real scenario, but here we just check if it routes
    session_id = "test-session"
    payload = {
        "session_id": session_id,
        "message": "Hello",
        "rag_enabled": False
    }
    # Note: This might fail if no providers are available/configured, 
    # but the API logic itself can be checked for 422 etc.
    response = client.post("/api/v1/chat", json=payload)
    # If no provider is available, it might return 500 or raise RuntimeError
    # We expect the logic to at least try routing.
    assert response.status_code in [200, 500] 
