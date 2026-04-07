"""
Write your tests here.
You should add and adjust tests at least for the endpoints you implement.
e.g. check responses, potential errors etc.
"""

from starlette.testclient import TestClient


def test_health(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200


def test_successful_call(client: TestClient):
    response = client.post(
        "/api/predict",
        json={"id": 1234, "recency_7": 1, "frequency_7": 1, "monetary_7": 8.5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1234
    assert isinstance(body["monetary_30"], (int, float))
    assert body["monetary_30"] >= 0


def test_predict_rejects_invalid_recency(client: TestClient):
    response = client.post(
        "/api/predict",
        json={"id": 1, "recency_7": 0, "frequency_7": 1, "monetary_7": 1.0},
    )
    assert response.status_code == 422


def test_predict_rejects_negative_monetary(client: TestClient):
    response = client.post(
        "/api/predict",
        json={"id": 1, "recency_7": 1, "frequency_7": 1, "monetary_7": -1.0},
    )
    assert response.status_code == 422


def test_requests_are_tracked(client: TestClient):
    passenger_id = 999
    for _ in range(3):
        resp = client.post(
            "/api/predict",
            json={"id": passenger_id, "recency_7": 2, "frequency_7": 1, "monetary_7": 1.0},
        )
        assert resp.status_code == 200

    count_resp = client.get(f"/api/requests/{passenger_id}")
    assert count_resp.status_code == 200
    body = count_resp.json()
    assert body["id"] == passenger_id
    assert body["count"] >= 3
