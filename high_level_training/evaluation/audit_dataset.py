"""Kiểm tra dataset trước khi train model mức cao.

Công cụ kiểm tra tên cột, kiểu số, nhãn LOW/MEDIUM/HIGH, event bị trùng,
snapshot không có event tương ứng và số lượng camera. Nếu còn lỗi, dataset
chưa nên được đưa vào bước train.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from backend.app.risk.dataset_exporter import EVENT_COLUMNS, FEATURE_COLUMNS, SNAPSHOT_COLUMNS


RISK_LABELS = {"LOW", "MEDIUM", "HIGH"}
INTEGER_FEATURES = {
    "reentry_count",
    "direction_change_count",
    "people_in_zone_peak",
    "observation_count",
}


def audit_dataset(events_path: Path, snapshots_path: Path) -> dict[str, Any]:
    """Kiểm tra hai file CSV và trả danh sách lỗi, cảnh báo cùng thống kê."""

    # BƯỚC 1: Đọc CSV và kiểm tra các cột bắt buộc.
    events = _read_csv(events_path, EVENT_COLUMNS)
    snapshots = _read_csv(snapshots_path, SNAPSHOT_COLUMNS)
    errors: list[str] = []
    warnings: list[str] = []

    # BƯỚC 2: Kiểm tra từng event và ghi nhớ camera của event đó.
    event_ids: set[str] = set()
    camera_by_event: dict[str, str] = {}
    duplicate_event_ids: set[str] = set()
    for line_number, row in enumerate(events, start=2):
        event_id = row["event_id"].strip()
        camera_id = row["camera_id"].strip()
        if not event_id:
            errors.append(f"events dòng {line_number}: event_id trống")
        elif event_id in event_ids:
            duplicate_event_ids.add(event_id)
        event_ids.add(event_id)
        camera_by_event[event_id] = camera_id
        _validate_common_row(row, line_number, "events", errors)
        _validate_label(row, line_number, "events", errors)

    if duplicate_event_ids:
        errors.append(f"event_id bị trùng trong events: {sorted(duplicate_event_ids)}")

    # BƯỚC 3: Kiểm tra snapshot có liên kết đúng event và đúng camera không.
    orphan_snapshot_ids: set[str] = set()
    for line_number, row in enumerate(snapshots, start=2):
        event_id = row["event_id"].strip()
        snapshot_camera_id = row["camera_id"].strip()
        if event_id not in event_ids:
            orphan_snapshot_ids.add(event_id)
        elif not snapshot_camera_id:
            errors.append(f"snapshots dòng {line_number}: camera_id trống")
        elif snapshot_camera_id != camera_by_event[event_id]:
            errors.append(
                f"snapshots dòng {line_number}: camera_id không khớp event {event_id}"
            )
        _validate_common_row(row, line_number, "snapshots", errors)
        _validate_label(row, line_number, "snapshots", errors)
    if orphan_snapshot_ids:
        errors.append(f"snapshot không có event tương ứng: {sorted(orphan_snapshot_ids)}")

    # BƯỚC 4: Thống kê số camera và phân bố nhãn để biết dataset đã đủ train chưa.
    camera_ids = {row["camera_id"].strip() for row in events if row["camera_id"].strip()}
    if len(camera_ids) < 3:
        warnings.append("có ít hơn 3 camera; chưa đủ để chia train/validation/test theo camera")
    if not events:
        warnings.append("dataset chưa có sự kiện")

    event_label_counts = Counter(_normalized_label(row) for row in events)
    snapshot_label_counts = Counter(_normalized_label(row) for row in snapshots)
    return {
        "events": len(events),
        "snapshots": len(snapshots),
        "cameras": len(camera_ids),
        "videos": len({row["video_id"].strip() for row in events if row["video_id"].strip()}),
        "event_label_counts": dict(sorted(event_label_counts.items())),
        "snapshot_label_counts": dict(sorted(snapshot_label_counts.items())),
        "errors": errors,
        "warnings": warnings,
        "ready_for_training": not errors
        and len(camera_ids) >= 3
        and all(snapshot_label_counts.get(label, 0) > 0 for label in RISK_LABELS),
    }


def _read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"không tìm thấy CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = set(expected_columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} thiếu cột: {sorted(missing)}")
        return list(reader)


def _validate_common_row(
    row: dict[str, str],
    line_number: int,
    source_name: str,
    errors: list[str],
) -> None:
    for feature in FEATURE_COLUMNS:
        raw_value = row[feature].strip()
        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"{source_name} dòng {line_number}: {feature} không phải số")
            continue
        if not math.isfinite(value):
            errors.append(f"{source_name} dòng {line_number}: {feature} không hữu hạn")
        elif value < 0:
            errors.append(f"{source_name} dòng {line_number}: {feature} không được âm")
        elif feature in INTEGER_FEATURES and not value.is_integer():
            errors.append(f"{source_name} dòng {line_number}: {feature} phải là số nguyên")

    try:
        confidence = float(row["detection_confidence_avg"])
        if not 0.0 <= confidence <= 1.0:
            errors.append(
                f"{source_name} dòng {line_number}: detection_confidence_avg phải trong [0, 1]"
            )
    except ValueError:
        pass


def _validate_label(
    row: dict[str, str],
    line_number: int,
    source_name: str,
    errors: list[str],
) -> None:
    label = _normalized_label(row)
    if label != "UNLABELED" and label not in RISK_LABELS:
        errors.append(f"{source_name} dòng {line_number}: nhãn không hợp lệ {row['label']!r}")


def _normalized_label(row: dict[str, str]) -> str:
    return row["label"].strip().upper() or "UNLABELED"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng dataset xử lý mức cao.")
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="File JSON để lưu báo cáo.")
    args = parser.parse_args()

    try:
        report = audit_dataset(args.events, args.snapshots)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Kiểm tra dataset thất bại: {exc}")
        return 1

    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
