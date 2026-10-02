"""Integration tests for FastAPI endpoints, database persistence, and latency benchmarks."""

import io
import time

import numpy as np
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    """Verifies that root endpoint returns operational metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/docs" in data["docs"]


def test_health_endpoint() -> None:
    """Verifies system health check, database connection, and loaded models."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "connected" in data["database"]
    assert len(data["loaded_models"]) >= 4
    assert "churn_lightgbm_v1" in data["loaded_models"]
    assert "clv_ridge_regressor_v1" in data["loaded_models"]


def test_models_metadata_endpoint() -> None:
    """Verifies retrieval of model hyperparameters, validation metrics, and feature lists."""
    response = client.get("/models/metadata")
    assert response.status_code == 200
    data = response.json()

    # Check Churn model
    churn_meta = data["production_churn_model"]
    assert churn_meta["model_name"] == "LightGBM Classifier"
    assert churn_meta["test_roc_auc"] >= 0.80
    assert churn_meta["test_recall"] >= 0.70

    # Check CLV model
    clv_meta = data["clv_regression_model"]
    assert clv_meta["model_name"] == "Ridge Regressor"
    assert clv_meta["test_r2"] > 0.50

    # Check Segmentation
    seg_meta = data["segmentation_model"]
    assert seg_meta["k_clusters"] == 4
    assert len(seg_meta["personas"]) == 4

    # Check Anomaly
    ae_meta = data["anomaly_autoencoder"]
    assert "threshold_p95" in ae_meta
    assert ae_meta["threshold_p95"] > 0.0

    # Check Features
    assert data["feature_count"] == 33
    assert len(data["active_features"]) == 33


def test_predict_single_customer() -> None:
    """Verifies single-customer multi-model prediction and narrative generation."""
    payload = {
        "customer_id": 14057,
        "recency_days": 18.0,
        "frequency": 12.0,
        "total_spend": 3200.50,
        "avg_basket_size": 25.0,
        "return_line_rate": 0.02,
        "is_uk": 1,
    }
    response = client.post("/predict/customer", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == 14057
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["is_churn"], bool)
    assert data["predicted_clv_90d"] >= 0.0
    assert data["risk_tier"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert data["segment_name"] in [
        "VIP Champions",
        "Loyal Regulars",
        "High-Value Inactive",
        "At-Risk / Lapsed",
    ]
    assert isinstance(data["is_anomaly"], bool)
    assert data["anomaly_score"] >= 0.0
    assert len(data["plain_language_narrative"]) > 20
    assert len(data["top_risk_factors"]) > 0


def test_predict_customer_minimal_payload() -> None:
    """Verifies that missing derived features are smoothly filled without error."""
    payload = {
        "recency_days": 120.0,
        "frequency": 1.0,
        "total_spend": 35.0,
    }
    response = client.post("/predict/customer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_probability"] >= 0.50  # Lapsed single-order customer should be high churn
    assert data["risk_tier"] in ["Medium Risk", "High Risk"]


def test_predict_customer_validation_error() -> None:
    """Verifies Pydantic validation rejects negative recency or malformed fields."""
    invalid_payload = {
        "recency_days": -10.0,  # Invalid: ge=0.0
        "frequency": 2.0,
        "total_spend": 100.0,
    }
    response = client.post("/predict/customer", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_batch_csv() -> None:
    """Verifies CSV upload batch scoring and summary generation."""
    csv_content = (
        "customer_id,recency_days,frequency,total_spend\n"
        "99101,15.0,8.0,2400.0\n"
        "99102,180.0,1.0,25.0\n"
        "99103,45.0,3.0,450.0\n"
    )
    files = {"file": ("batch_sample.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["total_records"] == 3
    assert 0.0 <= data["churn_rate"] <= 1.0
    assert data["mean_predicted_clv"] >= 0.0
    assert len(data["predictions"]) == 3
    assert data["download_csv_payload"] is not None
    assert "churn_probability" in data["download_csv_payload"]


def test_predict_batch_invalid_file() -> None:
    """Verifies non-CSV file uploads are rejected with 400 Bad Request."""
    files = {"file": ("bad_file.json", io.BytesIO(b'{"key": "value"}'), "application/json")}
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 400
    assert "format" in response.json()["detail"].lower()


def test_get_customer_existing() -> None:
    """Verifies retrieval of persisted customer details, segments, predictions, and RFM."""
    response = client.get("/customers/14057")
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == 14057
    assert data["country"] is not None
    assert data["segment"] is not None
    assert data["segment"]["segment_name"] in [
        "VIP Champions",
        "Loyal Regulars",
        "High-Value Inactive",
        "At-Risk / Lapsed",
    ]
    assert data["latest_prediction"] is not None
    assert data["latest_prediction"]["predicted_clv"] >= 0.0
    assert data["historical_predictions_count"] >= 1
    # Check RFM metrics
    if data["rfm_metrics"] is not None:
        assert data["rfm_metrics"]["recency_days"] >= 0.0
        assert data["rfm_metrics"]["frequency"] >= 1.0


def test_get_customer_not_found() -> None:
    """Verifies non-existent customer returns 404."""
    response = client.get("/customers/99999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_list_customers_pagination() -> None:
    """Verifies customer listing and pagination query parameters."""
    response = client.get("/customers?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 10
    assert "customer_id" in data[0]


def test_predict_customer_latency_p95() -> None:
    """Validates that p95 response time for /predict/customer is < 400ms."""
    payload = {
        "customer_id": 14057,
        "recency_days": 20.0,
        "frequency": 5.0,
        "total_spend": 850.0,
    }

    # Warmup
    _ = client.post("/predict/customer", json=payload)

    latencies_ms = []
    iterations = 30
    for _ in range(iterations):
        t0 = time.perf_counter()
        resp = client.post("/predict/customer", json=payload)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        latencies_ms.append((t1 - t0) * 1000.0)

    p95 = float(np.percentile(latencies_ms, 95))
    mean_lat = float(np.mean(latencies_ms))
    print(f"\n[LATENCY BENCHMARK] Mean: {mean_lat:.2f}ms, P95: {p95:.2f}ms across {iterations} calls")

    assert p95 < 400.0, f"P95 latency of {p95:.2f}ms exceeded 400ms SLA target"
