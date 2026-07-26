"""Áp dụng kết quả con người đã duyệt vào một bản dataset mới.

KEEP_HIGH giữ nguyên nhãn HIGH.
CHANGE_TO_MEDIUM đổi HIGH thành MEDIUM.
EXCLUDE_EVENT loại toàn bộ event và các snapshot thuộc event đó.
CSV nguồn luôn được giữ nguyên để có thể quay lại khi cần.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.app.risk.dataset_exporter import EVENT_COLUMNS, SNAPSHOT_COLUMNS


REVIEW_DECISIONS = {"KEEP_HIGH", "CHANGE_TO_MEDIUM", "EXCLUDE_EVENT"}


def apply_label_review(
    events_path: Path,
    snapshots_path: Path,
    review_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Đọc quyết định duyệt rồi tạo lại hai CSV event và snapshot."""

    # BƯỚC 1: Đọc dữ liệu gốc và các quyết định người dùng đã nhập.
    events = _read_rows(events_path, EVENT_COLUMNS)
    snapshots = _read_rows(snapshots_path, SNAPSHOT_COLUMNS)
    decisions = _read_decisions(review_path)
    event_ids = {row["event_id"].strip() for row in events}
    unknown_event_ids = set(decisions) - event_ids
    if unknown_event_ids:
        raise ValueError(f"review chứa event_id không tồn tại: {sorted(unknown_event_ids)}")

    # BƯỚC 2: Áp dụng quyết định cho từng dòng event.
    output_events: list[dict[str, str]] = []
    changed_events = 0
    excluded_events = 0
    for row in events:
        event_id = row["event_id"].strip()
        review = decisions.get(event_id)
        if review is None:
            output_events.append(row)
            continue
        decision, review_note = review
        if decision == "EXCLUDE_EVENT":
            excluded_events += 1
            continue
        if decision == "CHANGE_TO_MEDIUM" and row["label"].strip().upper() == "HIGH":
            row["label"] = "MEDIUM"
            changed_events += 1
        row["annotator_note"] = _append_review_note(
            row["annotator_note"],
            decision,
            review_note,
        )
        output_events.append(row)

    # BƯỚC 3: Áp dụng cùng quyết định cho mọi snapshot thuộc event.
    output_snapshots: list[dict[str, str]] = []
    changed_snapshots = 0
    excluded_snapshots = 0
    for row in snapshots:
        event_id = row["event_id"].strip()
        review = decisions.get(event_id)
        if review is None:
            output_snapshots.append(row)
            continue
        decision, review_note = review
        if decision == "EXCLUDE_EVENT":
            excluded_snapshots += 1
            continue
        if decision == "CHANGE_TO_MEDIUM" and row["label"].strip().upper() == "HIGH":
            row["label"] = "MEDIUM"
            changed_snapshots += 1
        row["annotator_note"] = _append_review_note(
            row["annotator_note"],
            decision,
            review_note,
        )
        output_snapshots.append(row)

    # BƯỚC 4: Ghi kết quả sang thư mục mới và tạo báo cáo số lượng thay đổi.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_output = output_dir / "high_level_events.csv"
    snapshots_output = output_dir / "high_level_snapshots.csv"
    _write_rows(events_output, EVENT_COLUMNS, output_events)
    _write_rows(snapshots_output, SNAPSHOT_COLUMNS, output_snapshots)

    report: dict[str, Any] = {
        "reviewed_events": len(decisions),
        "decision_counts": dict(
            sorted(Counter(decision for decision, _ in decisions.values()).items())
        ),
        "changed_events_high_to_medium": changed_events,
        "changed_snapshots_high_to_medium": changed_snapshots,
        "excluded_events": excluded_events,
        "excluded_snapshots": excluded_snapshots,
        "remaining_events": len(output_events),
        "remaining_snapshots": len(output_snapshots),
        "events_output": str(events_output),
        "snapshots_output": str(snapshots_output),
    }
    report_path = output_dir / "review_application_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def _read_decisions(path: Path) -> dict[str, tuple[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"event_id", "review_decision", "review_note"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"review CSV thiếu cột: {sorted(missing)}")

        decisions: dict[str, tuple[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            event_id = row["event_id"].strip()
            decision = row["review_decision"].strip().upper()
            if not event_id:
                raise ValueError(f"review dòng {line_number}: event_id trống")
            if event_id in decisions:
                raise ValueError(f"review dòng {line_number}: event_id bị trùng")
            if decision not in REVIEW_DECISIONS:
                raise ValueError(
                    f"review dòng {line_number}: review_decision phải thuộc "
                    f"{sorted(REVIEW_DECISIONS)}"
                )
            decisions[event_id] = (decision, row["review_note"].strip())
    if not decisions:
        raise ValueError("review CSV không có quyết định nào")
    return decisions


def _append_review_note(original: str, decision: str, review_note: str) -> str:
    suffix = f"Duyệt thủ công: {decision}"
    if review_note:
        suffix += f"; {review_note}"
    return f"{original.strip()} | {suffix}" if original.strip() else suffix


def _read_rows(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = set(expected_columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{Path(path).name} thiếu cột: {sorted(missing)}")
        return list(reader)


def _write_rows(
    path: Path,
    columns: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Áp dụng quyết định duyệt nhãn vào một bản dataset mới."
    )
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        report = apply_label_review(
            args.events,
            args.snapshots,
            args.review,
            args.output,
        )
    except (OSError, ValueError) as exc:
        print(f"Apply review failed: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
