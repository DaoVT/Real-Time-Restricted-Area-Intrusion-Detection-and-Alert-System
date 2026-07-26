from __future__ import annotations

import csv
import warnings

import joblib

from backend.app.risk.dataset_exporter import SNAPSHOT_COLUMNS
from high_level_training.evaluation.analyze_model_errors import analyze_model_errors
from high_level_training.evaluation.generate_evaluation_report import (
    generate_evaluation_report,
)
from high_level_training.modeling.train_random_forest import (
    MODEL_FEATURES,
    train_random_forest,
)


def test_train_random_forest_keeps_camera_groups_isolated(tmp_path):
    snapshots_path = tmp_path / "high_level_snapshots.csv"
    labels = ("LOW", "MEDIUM", "HIGH")
    with snapshots_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SNAPSHOT_COLUMNS)
        writer.writeheader()
        for camera_index in range(9):
            for label_index, label in enumerate(labels):
                event_id = f"camera-{camera_index}-event-{label.lower()}"
                for snapshot_index in range(2):
                    # Dữ liệu tổng hợp chỉ dùng kiểm thử pipeline, không được lưu làm model thật.
                    risk_value = float(label_index + 1) + snapshot_index * 0.05
                    row = {
                        "event_id": event_id,
                        "video_id": f"video-{camera_index}",
                        "camera_id": f"CAM_{camera_index}",
                        "track_id": label_index + 1,
                        "video_time_sec": snapshot_index + 1,
                        "elapsed_sec": snapshot_index + 1,
                        "inside_duration_sec": risk_value,
                        "average_speed_norm": risk_value / 10,
                        "max_speed_norm": risk_value / 8,
                        "max_zone_depth_norm": risk_value / 20,
                        "stopped_duration_sec": risk_value / 5,
                        "reentry_count": label_index,
                        "direction_change_count": label_index,
                        "people_in_zone_peak": label_index + 1,
                        "observation_count": 10,
                        "detection_confidence_avg": 0.9,
                        "label": label,
                        "annotator_note": "synthetic test row",
                    }
                    writer.writerow(row)

    output_dir = tmp_path / "trained"
    metrics = train_random_forest(snapshots_path, output_dir, random_state=7)

    assert (output_dir / "risk_model.joblib").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "confusion_matrix.csv").exists()
    assert (output_dir / "feature_importance.csv").exists()
    assert metrics["skipped_unlabeled_rows"] == 0
    assert all(
        set(label_counts) == set(labels)
        for label_counts in metrics["split_label_counts"].values()
    )
    assert metrics["split_camera_counts"]["train"] >= 4
    assert metrics["split_camera_counts"]["validation"] >= 2
    assert metrics["split_camera_counts"]["test"] >= 2

    # Joblib hiện phát cảnh báo tương thích với NumPy 2.5 cho từng cây; artifact vẫn được đọc đúng.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        artifact = joblib.load(output_dir / "risk_model.joblib")
    assert artifact["feature_columns"] == list(MODEL_FEATURES)

    with (output_dir / "split_assignments.csv").open(encoding="utf-8") as file:
        assignments = list(csv.DictReader(file))
    cameras_by_split = {
        split: {row["camera_id"] for row in assignments if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert cameras_by_split["train"].isdisjoint(cameras_by_split["validation"])
    assert cameras_by_split["train"].isdisjoint(cameras_by_split["test"])
    assert cameras_by_split["validation"].isdisjoint(cameras_by_split["test"])

    evaluation_dir = tmp_path / "evaluation"
    evaluation = generate_evaluation_report(
        snapshots_path,
        output_dir / "risk_model.joblib",
        output_dir / "split_assignments.csv",
        evaluation_dir,
    )
    assert evaluation["evaluated_snapshots"] == metrics["split_sizes"]["test"]
    assert set(evaluation["average_precision"]) == set(labels)
    assert set(evaluation["roc_auc"]) == set(labels)
    assert (evaluation_dir / "test_predictions.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    for image_name in (
        "confusion_matrix.png",
        "feature_importance.png",
        "precision_recall_curve.png",
        "roc_curve.png",
        "confidence_threshold_curve.png",
        "class_distribution.png",
    ):
        assert (evaluation_dir / image_name).read_bytes().startswith(b"\x89PNG")

    analysis_dir = tmp_path / "analysis"
    report = analyze_model_errors(
        snapshots_path,
        _write_events_for_snapshots(tmp_path, snapshots_path),
        output_dir / "risk_model.joblib",
        output_dir / "split_assignments.csv",
        analysis_dir,
    )
    assert report["evaluated_snapshots"] > 0
    assert (analysis_dir / "misclassified_snapshots.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (analysis_dir / "misclassified_events.csv").exists()
    assert (analysis_dir / "high_to_medium_review.csv").exists()


def _write_events_for_snapshots(tmp_path, snapshots_path):
    from backend.app.risk.dataset_exporter import EVENT_COLUMNS

    events_path = tmp_path / "high_level_events.csv"
    with snapshots_path.open(encoding="utf-8") as snapshots_file:
        snapshots = list(csv.DictReader(snapshots_file))
    seen: set[str] = set()
    with events_path.open("w", newline="", encoding="utf-8") as events_file:
        writer = csv.DictWriter(events_file, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        for row in snapshots:
            if row["event_id"] in seen:
                continue
            seen.add(row["event_id"])
            writer.writerow(
                {
                    "event_id": row["event_id"],
                    "video_id": row["video_id"],
                    "camera_id": row["camera_id"],
                    "track_id": row["track_id"],
                    "entry_time_sec": 0,
                    "confirmed_time_sec": 0.5,
                    "end_time_sec": 2,
                    "event_duration_sec": 2,
                    "evidence_path": str(tmp_path / f"{row['event_id']}.jpg"),
                    **{
                        column: row[column]
                        for column in EVENT_COLUMNS
                        if column in row
                    },
                }
            )
    return events_path
