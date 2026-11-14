from fastapi.testclient import TestClient

from src.api.app import app


def test_health():
    client = TestClient(app)
    assert client.get("/").json() == {"status": "ok"}
