# service/tests/test_api.py
from fastapi.testclient import TestClient
import os
from app.main import app

def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_requires_key():
    client = TestClient(app)
    r = client.post("/predict", json={"texts":["test"]})
    assert r.status_code in (401, 500)
