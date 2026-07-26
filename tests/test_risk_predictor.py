from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from backend.app.risk import FeatureVector, RiskPredictor


class FakeRiskModel:
    classes_ = np.array(["HIGH", "LOW", "MEDIUM"])

    def predict(self, features):
        assert len(features[0]) == 8
        return np.array(["HIGH"])

    def predict_proba(self, features):
        assert len(features[0]) == 8
        return np.array([[0.8, 0.05, 0.15]])


def test_risk_predictor_loads_artifact_and_returns_websocket_data(tmp_path: Path):
    model_path = tmp_path / "risk_model.joblib"
    joblib.dump(
        {
            "model": FakeRiskModel(),
            "feature_columns": [
                "inside_duration_sec",
                "average_speed_norm",
                "max_speed_norm",
                "max_zone_depth_norm",
                "stopped_duration_sec",
                "reentry_count",
                "direction_change_count",
                "people_in_zone_peak",
            ],
            "trained_at_utc": "2026-07-24T00:00:00+00:00",
        },
        model_path,
    )
    snapshot = FeatureVector(
        event_id="event-1",
        track_id=7,
        entry_time_sec=0.0,
        confirmed_time_sec=0.5,
        sample_time_sec=3.0,
        elapsed_sec=3.0,
        inside_duration_sec=3.0,
        average_speed_norm=0.02,
        max_speed_norm=0.12,
        max_zone_depth_norm=0.11,
        stopped_duration_sec=0.5,
        reentry_count=0,
        direction_change_count=2,
        people_in_zone_peak=3,
        observation_count=20,
        detection_confidence_avg=0.9,
    )

    result = RiskPredictor(model_path).predict(snapshot)

    assert result["available"] is True
    assert result["event_id"] == "event-1"
    assert result["track_id"] == 7
    assert result["level"] == "HIGH"
    assert result["probabilities"] == {
        "LOW": 0.05,
        "MEDIUM": 0.15,
        "HIGH": 0.8,
    }
