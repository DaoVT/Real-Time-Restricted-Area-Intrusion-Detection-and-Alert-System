"""Tìm những dự đoán sai để người dùng xem lại.

Luồng xử lý:
1. Đọc model, snapshot, event và phép chia train/validation/test.
2. Chỉ chạy dự đoán trên tập cần kiểm tra.
3. So sánh nhãn thật với nhãn model dự đoán.
4. Xuất các dòng sai và ảnh bằng chứng để duyệt lại nhãn.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib

from backend.app.risk.dataset_exporter import EVENT_COLUMNS, SNAPSHOT_COLUMNS
from high_level_training.modeling.train_random_forest import RISK_LABELS


ERROR_COLUMNS = [
    "event_id",
    "video_id",
    "camera_id",
    "track_id",
    "video_time_sec",
    "actual_label",
    "predicted_label",
    "probability_low",
    "probability_medium",
    "probability_high",
    "evidence_path",
    "inside_duration_sec",
    "average_speed_norm",
    "max_speed_norm",
    "max_zone_depth_norm",
    "stopped_duration_sec",
    "reentry_count",
    "direction_change_count",
    "people_in_zone_peak",
    "observation_count",
    "detection_confidence_avg",
    "annotator_note",
]

EVENT_REVIEW_COLUMNS = [
    "event_id",
    "video_id",
    "camera_id",
    "track_id",
    "actual_labels",
    "predicted_labels",
    "misclassified_snapshot_count",
    "high_to_medium_count",
    "max_probability_high",
    "evidence_path",
]

HIGH_REVIEW_COLUMNS = [
    "review_index",
    "event_id",
    "video_id",
    "camera_id",
    "track_id",
    "event_final_label",
    "snapshot_label",
    "model_prediction",
    "high_to_medium_snapshot_count",
    "average_probability_medium",
    "average_probability_high",
    "max_probability_high",
    "inside_duration_sec",
    "average_speed_norm",
    "max_speed_norm",
    "max_zone_depth_norm",
    "stopped_duration_sec",
    "reentry_count",
    "direction_change_count",
    "people_in_zone_peak",
    "detection_confidence_avg",
    "automatic_note",
    "evidence_path",
    "review_evidence_path",
    "review_decision",
    "review_note",
]


def analyze_model_errors(
    snapshots_path: Path,
    events_path: Path,
    model_path: Path,
    assignments_path: Path,
    output_dir: Path,
    split: str = "test",
) -> dict[str, Any]:
    """Tìm các snapshot bị dự đoán sai trong một tập dữ liệu đã chọn."""

    # Joblib hiện phát cảnh báo tương thích khi khôi phục từng cây với NumPy 2.5;
    # cảnh báo này không làm thay đổi dữ liệu hoặc kết quả dự đoán của artifact.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        artifact = joblib.load(model_path)
    model = artifact.get("model")
    feature_columns = artifact.get("feature_columns")
    if model is None or not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("artifact model thiếu model hoặc feature_columns")

    snapshots = _read_rows(snapshots_path, SNAPSHOT_COLUMNS)
    events = _read_rows(events_path, EVENT_COLUMNS)
    split_by_event = _read_assignments(assignments_path)
    event_by_id = {row["event_id"].strip(): row for row in events}

    selected = [
        row
        for row in snapshots
        if split_by_event.get(row["event_id"].strip()) == split
    ]
    if not selected:
        raise ValueError(f"không có snapshot thuộc split {split!r}")

    features = [
        [_read_number(row, column) for column in feature_columns]
        for row in selected
    ]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    class_indexes = {
        str(label): index
        for index, label in enumerate(model.classes_)
    }

    errors: list[dict[str, Any]] = []
    confusion_counts: Counter[str] = Counter()
    for row, predicted, probability_values in zip(
        selected,
        predictions,
        probabilities,
    ):
        actual = row["label"].strip().upper()
        predicted_label = str(predicted)
        confusion_counts[f"{actual}->{predicted_label}"] += 1
        if predicted_label == actual:
            continue

        event_id = row["event_id"].strip()
        event = event_by_id.get(event_id)
        if event is None:
            raise ValueError(f"snapshot không có event tương ứng: {event_id}")
        errors.append(
            {
                "event_id": event_id,
                "video_id": row["video_id"],
                "camera_id": row["camera_id"],
                "track_id": row["track_id"],
                "video_time_sec": row["video_time_sec"],
                "actual_label": actual,
                "predicted_label": predicted_label,
                "probability_low": _class_probability(
                    probability_values,
                    class_indexes,
                    "LOW",
                ),
                "probability_medium": _class_probability(
                    probability_values,
                    class_indexes,
                    "MEDIUM",
                ),
                "probability_high": _class_probability(
                    probability_values,
                    class_indexes,
                    "HIGH",
                ),
                "evidence_path": event["evidence_path"],
                **{
                    column: row[column]
                    for column in ERROR_COLUMNS
                    if column in row
                    and column
                    not in {
                        "event_id",
                        "video_id",
                        "camera_id",
                        "track_id",
                        "video_time_sec",
                        "annotator_note",
                    }
                },
                "annotator_note": row["annotator_note"],
            }
        )

    # Đưa lỗi HIGH bị đoán thành MEDIUM lên đầu vì đây là cảnh báo nguy hiểm bị bỏ sót.
    errors.sort(
        key=lambda row: (
            row["actual_label"] != "HIGH",
            row["predicted_label"] != "MEDIUM",
            -float(row["probability_high"]),
        )
    )
    event_reviews = _aggregate_events(errors)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "misclassified_snapshots.csv"
    events_path_output = output_dir / "misclassified_events.csv"
    _write_rows(errors_path, ERROR_COLUMNS, errors)
    _write_rows(events_path_output, EVENT_REVIEW_COLUMNS, event_reviews)
    high_review_rows = _build_high_review_rows(
        errors,
        event_by_id,
        output_dir / "review_evidence",
    )
    high_review_path = output_dir / "high_to_medium_review.csv"
    _write_rows(high_review_path, HIGH_REVIEW_COLUMNS, high_review_rows)

    error_type_counts = Counter(
        f"{row['actual_label']}->{row['predicted_label']}"
        for row in errors
    )
    missing_evidence = sum(
        not Path(row["evidence_path"]).is_file()
        for row in event_reviews
        if row["evidence_path"]
    )
    report: dict[str, Any] = {
        "split": split,
        "evaluated_snapshots": len(selected),
        "misclassified_snapshots": len(errors),
        "misclassified_events": len(event_reviews),
        "high_to_medium_review_events": len(high_review_rows),
        "error_type_counts": dict(sorted(error_type_counts.items())),
        "confusion_counts": dict(sorted(confusion_counts.items())),
        "missing_evidence_files": missing_evidence,
        "misclassified_snapshots_path": str(errors_path),
        "misclassified_events_path": str(events_path_output),
        "high_to_medium_review_path": str(high_review_path),
    }
    report_path = output_dir / "error_analysis_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def _build_high_review_rows(
    errors: list[dict[str, Any]],
    event_by_id: dict[str, dict[str, str]],
    evidence_output_dir: Path,
) -> list[dict[str, Any]]:
    """Gộp các snapshot sai theo event để mỗi sự kiện chỉ cần duyệt một lần."""

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in errors:
        if row["actual_label"] == "HIGH" and row["predicted_label"] == "MEDIUM":
            by_event[str(row["event_id"])].append(row)

    evidence_output_dir.mkdir(parents=True, exist_ok=True)
    review_rows: list[dict[str, Any]] = []
    sorted_events = sorted(
        by_event.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for review_index, (event_id, rows) in enumerate(sorted_events, start=1):
        event = event_by_id[event_id]
        source_evidence = Path(event["evidence_path"])
        review_evidence = ""
        if source_evidence.is_file():
            destination = (
                evidence_output_dir
                / f"{review_index:03d}__{event_id}{source_evidence.suffix.lower()}"
            )
            shutil.copy2(source_evidence, destination)
            review_evidence = str(destination)

        medium_probabilities = [float(row["probability_medium"]) for row in rows]
        high_probabilities = [float(row["probability_high"]) for row in rows]
        review_rows.append(
            {
                "review_index": review_index,
                "event_id": event_id,
                "video_id": event["video_id"],
                "camera_id": event["camera_id"],
                "track_id": event["track_id"],
                "event_final_label": event["label"],
                "snapshot_label": "HIGH",
                "model_prediction": "MEDIUM",
                "high_to_medium_snapshot_count": len(rows),
                "average_probability_medium": sum(medium_probabilities) / len(rows),
                "average_probability_high": sum(high_probabilities) / len(rows),
                "max_probability_high": max(high_probabilities),
                **{
                    column: event[column]
                    for column in (
                        "inside_duration_sec",
                        "average_speed_norm",
                        "max_speed_norm",
                        "max_zone_depth_norm",
                        "stopped_duration_sec",
                        "reentry_count",
                        "direction_change_count",
                        "people_in_zone_peak",
                        "detection_confidence_avg",
                    )
                },
                "automatic_note": event["annotator_note"],
                "evidence_path": event["evidence_path"],
                "review_evidence_path": review_evidence,
                # Người duyệt điền KEEP_HIGH, CHANGE_TO_MEDIUM hoặc EXCLUDE_EVENT.
                "review_decision": "",
                "review_note": "",
            }
        )
    return review_rows


def _aggregate_events(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in errors:
        by_event[str(row["event_id"])].append(row)

    reviews: list[dict[str, Any]] = []
    for event_id, rows in by_event.items():
        first = rows[0]
        reviews.append(
            {
                "event_id": event_id,
                "video_id": first["video_id"],
                "camera_id": first["camera_id"],
                "track_id": first["track_id"],
                "actual_labels": "|".join(sorted({str(row["actual_label"]) for row in rows})),
                "predicted_labels": "|".join(
                    sorted({str(row["predicted_label"]) for row in rows})
                ),
                "misclassified_snapshot_count": len(rows),
                "high_to_medium_count": sum(
                    row["actual_label"] == "HIGH"
                    and row["predicted_label"] == "MEDIUM"
                    for row in rows
                ),
                "max_probability_high": max(
                    float(row["probability_high"])
                    for row in rows
                ),
                "evidence_path": first["evidence_path"],
            }
        )
    reviews.sort(
        key=lambda row: (
            -int(row["high_to_medium_count"]),
            -int(row["misclassified_snapshot_count"]),
        )
    )
    return reviews


def _class_probability(
    values: Any,
    class_indexes: dict[str, int],
    label: str,
) -> float:
    index = class_indexes.get(label)
    return float(values[index]) if index is not None else 0.0


def _read_number(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{column} không phải giá trị số hợp lệ") from exc


def _read_rows(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = set(expected_columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{Path(path).name} thiếu cột: {sorted(missing)}")
        return list(reader)


def _read_assignments(path: Path) -> dict[str, str]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"event_id", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"split assignments thiếu cột: {sorted(missing)}")
        return {
            row["event_id"].strip(): row["split"].strip()
            for row in reader
        }


def _write_rows(
    path: Path,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> None:
    # UTF-8 BOM giúp Excel trên Windows hiển thị đúng ghi chú tiếng Việt.
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phân tích các snapshot bị model mức cao dự đoán sai."
    )
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    args = parser.parse_args()

    try:
        report = analyze_model_errors(
            args.snapshots,
            args.events,
            args.model,
            args.assignments,
            args.output,
            args.split,
        )
    except (OSError, ValueError) as exc:
        print(f"Error analysis failed: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
