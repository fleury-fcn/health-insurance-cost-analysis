import pytest
from fastapi.testclient import TestClient

from app.api import app

PROFILE = {"age": 45, "sex": "male", "bmi": 32.5, "children": 2, "smoker": "yes",
           "region": "southeast"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_positive_cost(client):
    body = client.post("/predict", json=PROFILE).json()
    assert body["predicted_charges"] > 0
    assert body["contributions"] is None


def test_smoker_costs_more_than_non_smoker(client):
    smoker = client.post("/predict", json=PROFILE).json()["predicted_charges"]
    non = client.post("/predict", json=PROFILE | {"smoker": "no"}).json()["predicted_charges"]
    assert smoker > non * 2


def test_explanation_adds_up(client):
    body = client.post("/predict?explain=true", json=PROFILE).json()
    total = body["base_value"] + sum(body["contributions"].values())
    assert total == pytest.approx(body["predicted_charges"], abs=1)


@pytest.mark.parametrize("bad", [{"age": 10}, {"smoker": "maybe"}, {"region": "mars"},
                                 {"bmi": -3}])
def test_invalid_input_is_rejected(client, bad):
    assert client.post("/predict", json=PROFILE | bad).status_code == 422


def test_swagger_and_model_card(client):
    assert client.get("/docs").status_code == 200
    assert "test_metrics" in client.get("/model").json()
