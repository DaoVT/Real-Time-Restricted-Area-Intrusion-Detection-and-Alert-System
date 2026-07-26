from __future__ import annotations

import csv

from backend.app.risk.dataset_exporter import EVENT_COLUMNS, SNAPSHOT_COLUMNS
from high_level_training.evaluation.audit_dataset import audit_dataset
from high_level_training.labeling.apply_label_review import apply_label_review
from high_level_training.labeling.auto_label_dataset import (
    auto_label_dataset,
    score_behavioral_policy_v2,
    score_risk,
)
from high_level_training.labeling.propagate_event_labels import propagate_event_labels


def _event_row(event_id: str, camera_id: str, label: str = "") -> dict[str, object]:
    return {
        "event_id": event_id,
        "video_id": f"video-{camera_id}",
        "camera_id": camera_id,
        "track_id": 1,
        "entry_time_sec": 0.0,
        "confirmed_time_sec": 0.5,
        "end_time_sec": 2.0,
        "event_duration_sec": 2.0,
        "evidence_path": "evidence.jpg",
        "inside_duration_sec": 2.0,
        "average_speed_norm": 0.01,
        "max_speed_norm": 0.02,
        "max_zone_depth_norm": 0.1,
        "stopped_duration_sec": 0.5,
        "reentry_count": 0,
        "direction_change_count": 1,
        "people_in_zone_peak": 1,
        "observation_count": 20,
        "detection_confidence_avg": 0.9,
        "label": label,
        "annotator_note": f"ghi chú {label}",
    }


def _snapshot_row(event_id: str, camera_id: str) -> dict[str, object]:
    event = _event_row(event_id, camera_id)
    return {
        column: (
            1.0
            if column == "video_time_sec"
            else 1.0
            if column == "elapsed_sec"
            else event[column]
        )
        for column in SNAPSHOT_COLUMNS
    }


def _write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_propagate_event_labels_creates_new_labeled_snapshot_file(tmp_path):
    events_path = tmp_path / "events.csv"
    snapshots_path = tmp_path / "snapshots.csv"
    destination = tmp_path / "snapshots_labeled.csv"
    _write_csv(events_path, EVENT_COLUMNS, [_event_row("event-1", "CAM_1", "HIGH")])
    _write_csv(snapshots_path, SNAPSHOT_COLUMNS, [_snapshot_row("event-1", "CAM_1")])

    summary = propagate_event_labels(events_path, snapshots_path, destination)

    assert summary == {"updated": 1, "unlabeled": 0, "orphaned": 0}
    with destination.open(encoding="utf-8") as file:
        [row] = list(csv.DictReader(file))
    assert row["label"] == "HIGH"
    assert row["annotator_note"] == "ghi chú HIGH"


def test_audit_dataset_reports_ready_dataset(tmp_path):
    labels = ("LOW", "MEDIUM", "HIGH")
    events = [
        _event_row(f"event-{index}", f"CAM_{index}", label)
        for index, label in enumerate(labels, start=1)
    ]
    snapshots = [
        {**_snapshot_row(row["event_id"], row["camera_id"]), "label": row["label"]}
        for row in events
    ]
    events_path = tmp_path / "events.csv"
    snapshots_path = tmp_path / "snapshots.csv"
    _write_csv(events_path, EVENT_COLUMNS, events)
    _write_csv(snapshots_path, SNAPSHOT_COLUMNS, snapshots)

    report = audit_dataset(events_path, snapshots_path)

    assert report["errors"] == []
    assert report["cameras"] == 3
    assert report["ready_for_training"] is True


def test_audit_dataset_detects_orphan_snapshot_and_invalid_confidence(tmp_path):
    events_path = tmp_path / "events.csv"
    snapshots_path = tmp_path / "snapshots.csv"
    event = _event_row("event-1", "CAM_1", "LOW")
    event["detection_confidence_avg"] = 1.5
    _write_csv(events_path, EVENT_COLUMNS, [event])
    _write_csv(snapshots_path, SNAPSHOT_COLUMNS, [_snapshot_row("missing-event", "CAM_1")])

    report = audit_dataset(events_path, snapshots_path)

    assert report["ready_for_training"] is False
    assert any("detection_confidence_avg" in error for error in report["errors"])
    assert any("snapshot không có event" in error for error in report["errors"])


def test_score_risk_returns_explainable_low_medium_and_high_labels():
    low = _snapshot_row("event-low", "CAM_1")
    low.update(
        inside_duration_sec="2",
        average_speed_norm="0.005",
        max_speed_norm="0.02",
        max_zone_depth_norm="0.02",
        stopped_duration_sec="0",
        reentry_count="0",
        direction_change_count="0",
        people_in_zone_peak="1",
    )
    medium = {**low, "inside_duration_sec": "10", "max_zone_depth_norm": "0.06", "people_in_zone_peak": "3"}
    high = {
        **low,
        "inside_duration_sec": "35",
        "max_zone_depth_norm": "0.12",
        "stopped_duration_sec": "21",
        "reentry_count": "1",
    }

    assert score_risk(low)[0] == "LOW"
    assert score_risk(medium)[0] == "MEDIUM"
    label, score, reasons = score_risk(high)
    assert label == "HIGH"
    assert score >= 7
    assert any("tái xâm nhập" in reason for reason in reasons)


def test_behavioral_policy_v2_requires_context_for_high_label():
    row = _snapshot_row("event-1", "CAM_1")
    row.update(
        inside_duration_sec="2.5",
        max_speed_norm="0.12",
        max_zone_depth_norm="0.13",
        stopped_duration_sec="0",
        reentry_count="0",
        people_in_zone_peak="3",
        direction_change_count="0",
    )
    assert score_behavioral_policy_v2(row)[0] == "MEDIUM"

    row.update(people_in_zone_peak="5")
    label, strong_count, reasons = score_behavioral_policy_v2(row)
    assert label == "HIGH"
    assert strong_count == 3
    assert "có ít nhất 5 người trong vùng" in reasons


def test_auto_label_dataset_repairs_snapshot_camera_id_and_preserves_sources(tmp_path):
    events_path = tmp_path / "events.csv"
    snapshots_path = tmp_path / "snapshots.csv"
    output_dir = tmp_path / "weak-labeled"
    event = _event_row("event-1", "CAM_1")
    snapshot = _snapshot_row("event-1", "")
    _write_csv(events_path, EVENT_COLUMNS, [event])
    _write_csv(snapshots_path, SNAPSHOT_COLUMNS, [snapshot])

    report = auto_label_dataset(events_path, snapshots_path, output_dir)

    assert report["repaired_snapshot_camera_ids"] == 1
    assert event["label"] == ""
    with (output_dir / "high_level_snapshots.csv").open(encoding="utf-8") as file:
        [labeled_snapshot] = list(csv.DictReader(file))
    assert (output_dir / "high_level_snapshots.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert labeled_snapshot["camera_id"] == "CAM_1"
    assert labeled_snapshot["label"] in {"LOW", "MEDIUM", "HIGH"}
    assert labeled_snapshot["annotator_note"].startswith("Nhãn tự động theo luật v1")


def test_audit_dataset_detects_missing_snapshot_camera_id(tmp_path):
    events_path = tmp_path / "events.csv"
    snapshots_path = tmp_path / "snapshots.csv"
    _write_csv(events_path, EVENT_COLUMNS, [_event_row("event-1", "CAM_1", "LOW")])
    _write_csv(
        snapshots_path,
        SNAPSHOT_COLUMNS,
        [{**_snapshot_row("event-1", ""), "label": "LOW"}],
    )

    report = audit_dataset(events_path, snapshots_path)

    assert report["ready_for_training"] is False
    assert any("camera_id trống" in error for error in report["errors"])


def test_apply_label_review_changes_only_high_rows_and_keeps_sources(tmp_path):
    events_path = tmp_path / "events.csv"
    snapshots_path = tmp_path / "snapshots.csv"
    review_path = tmp_path / "review.csv"
    output_dir = tmp_path / "reviewed"
    event = _event_row("event-1", "CAM_1", "HIGH")
    snapshots = [
        {**_snapshot_row("event-1", "CAM_1"), "label": "MEDIUM"},
        {**_snapshot_row("event-1", "CAM_1"), "label": "HIGH"},
    ]
    _write_csv(events_path, EVENT_COLUMNS, [event])
    _write_csv(snapshots_path, SNAPSHOT_COLUMNS, snapshots)
    with review_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["event_id", "review_decision", "review_note"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "event_id": "event-1",
                "review_decision": "CHANGE_TO_MEDIUM",
                "review_note": "chưa đủ dấu hiệu nguy hiểm cao",
            }
        )

    report = apply_label_review(
        events_path,
        snapshots_path,
        review_path,
        output_dir,
    )

    assert report["changed_events_high_to_medium"] == 1
    assert report["changed_snapshots_high_to_medium"] == 1
    with (output_dir / "high_level_snapshots.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert [row["label"] for row in rows] == ["MEDIUM", "MEDIUM"]
    assert event["label"] == "HIGH"
