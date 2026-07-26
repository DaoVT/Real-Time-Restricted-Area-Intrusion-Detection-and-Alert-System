"""Dùng model đã train để dự đoán mức rủi ro khi video đang chạy.

Mỗi FeatureVector được đổi thành 8 số theo đúng thứ tự lúc train.
Random Forest trả một nhãn LOW/MEDIUM/HIGH và xác suất của cả ba nhãn.
Kết quả sau đó được JobManager gửi lên giao diện qua WebSocket.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib

from .feature_extractor import FeatureVector


RISK_LABELS = ("LOW", "MEDIUM", "HIGH")


class RiskPredictor:
    """Mở Random Forest một lần rồi dùng lại cho mọi snapshot và mọi job."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"risk model not found: {self.model_path}")

        # Đọc gói model gồm Random Forest, tên các thuộc tính và thời điểm train.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Setting the shape on a NumPy array has been deprecated.*",
                category=DeprecationWarning,
            )
            artifact = joblib.load(self.model_path)

        self.model = artifact.get("model")
        self.feature_columns = artifact.get("feature_columns")
        self.trained_at_utc = artifact.get("trained_at_utc")
        if self.model is None or not isinstance(self.feature_columns, list):
            raise ValueError("risk model artifact thiếu model hoặc feature_columns")
        if not hasattr(self.model, "predict") or not hasattr(self.model, "predict_proba"):
            raise ValueError("risk model không hỗ trợ predict/predict_proba")
        classes = {str(label) for label in self.model.classes_}
        if classes != set(RISK_LABELS):
            raise ValueError(f"risk model có nhãn không hợp lệ: {sorted(classes)}")

        # Khóa này cho phép nhiều job dùng chung model nhưng chỉ dự đoán lần lượt,
        # tránh hai luồng cùng truy cập Random Forest tại một thời điểm.
        self._lock = threading.Lock()

    def predict(self, snapshot: FeatureVector) -> dict[str, Any]:
        """Dự đoán một snapshot và trả dữ liệu LOW/MEDIUM/HIGH cho WebSocket."""

        # Lấy đúng 8 thuộc tính và giữ nguyên thứ tự đã dùng khi train model.
        values = asdict(snapshot)
        try:
            features = [[float(values[column]) for column in self.feature_columns]]
        except KeyError as exc:
            raise ValueError(f"snapshot thiếu đặc trưng cho risk model: {exc.args[0]}") from exc

        # predict chọn nhãn cuối cùng; predict_proba trả xác suất của từng nhãn.
        with self._lock:
            prediction = str(self.model.predict(features)[0])
            probability_values = self.model.predict_proba(features)[0]
        probabilities = {
            str(label): float(probability_values[index])
            for index, label in enumerate(self.model.classes_)
        }
        return {
            "available": True,
            "event_id": snapshot.event_id,
            "track_id": snapshot.track_id,
            "video_time_sec": snapshot.sample_time_sec,
            "level": prediction,
            "probabilities": {
                label: probabilities.get(label, 0.0)
                for label in RISK_LABELS
            },
            "model_trained_at_utc": self.trained_at_utc,
        }
