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
    # Verify the response contract matches the README
    assert body["passenger_id"] == 1234
    assert isinstance(body["predicted_monetary_value_30"], (int, float))
    assert body["predicted_monetary_value_30"] >= 0
    assert "model_version" in body


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
    # Use a unique passenger_id isolated in the in-memory test DB
    passenger_id = 77777
    n_requests = 3
    for _ in range(n_requests):
        resp = client.post(
            "/api/predict",
            json={"id": passenger_id, "recency_7": 2, "frequency_7": 1, "monetary_7": 1.0},
        )
        assert resp.status_code == 200

    count_resp = client.get(f"/api/requests/{passenger_id}")
    assert count_resp.status_code == 200
    body = count_resp.json()
    assert body["id"] == passenger_id
    # Now exactly == n_requests because the in-memory DB is fresh per test session
    assert body["count"] == n_requests
