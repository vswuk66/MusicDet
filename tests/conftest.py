import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    from app import app
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """Get authentication headers with a valid token."""
    response = client.post(
        "/token",
        data={"username": "testuser", "password": "testpass"}
    )
    if response.status_code == 200:
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    return {}
