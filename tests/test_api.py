import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from api.main import app

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "vaga_texto": "Procuramos Python, SQL e AWS",
        "cv_texto": "3 anos com Python e AWS; Docker básico",
        "threshold": 0.6
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "classificacao_baseline" in data
    assert "skills_match" in data
