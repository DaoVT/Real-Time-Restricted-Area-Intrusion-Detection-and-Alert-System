"""Chép nhãn cuối của event sang các snapshot thuộc cùng event.

Ví dụ event A được duyệt là HIGH thì mọi snapshot có event_id A cũng nhận HIGH.
Cách này tạo nhanh một baseline theo sự kiện, nhưng không mô tả trường hợp
mức rủi ro thay đổi dần trong quá trình sự kiện diễn ra.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from backend.app.risk.dataset_exporter import EVENT_COLUMNS, SNAPSHOT_COLUMNS


RISK_LABELS = {"LOW", "MEDIUM", "HIGH"}


def propagate_event_labels(
    events_path: Path,
    snapshots_path: Path,
    destination_path: Path,
) -> dict[str, int]:
    """Tìm snapshot cùng event_id rồi chép nhãn và ghi chú của event sang."""

    # BƯỚC 1: Tạo bảng tra event_id -> nhãn.
    labels_by_event = _load_event_labels(events_path)
    snapshots_path = Path(snapshots_path)
    destination_path = Path(destination_path)
    if snapshots_path.resolve() == destination_path.resolve():
        raise ValueError("destination phải khác file snapshots gốc")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    updated = 0
    unlabeled = 0
    orphaned = 0
    # BƯỚC 2: Đọc từng snapshot, tìm event tương ứng và ghi sang file mới.
    with snapshots_path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        missing = set(SNAPSHOT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{snapshots_path.name} thiếu cột: {sorted(missing)}")

        with destination_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                event_id = row["event_id"].strip()
                event_label = labels_by_event.get(event_id)
                if event_label is None:
                    orphaned += 1
                elif not event_label[0]:
                    unlabeled += 1
                else:
                    row["label"], row["annotator_note"] = event_label
                    updated += 1
                writer.writerow(row)
    return {"updated": updated, "unlabeled": unlabeled, "orphaned": orphaned}


def _load_event_labels(path: Path) -> dict[str, tuple[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = set(EVENT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{Path(path).name} thiếu cột: {sorted(missing)}")

        labels: dict[str, tuple[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            event_id = row["event_id"].strip()
            if not event_id:
                raise ValueError(f"event_id trống tại dòng {line_number}")
            if event_id in labels:
                raise ValueError(f"event_id bị trùng tại dòng {line_number}: {event_id}")
            label = row["label"].strip().upper()
            if label and label not in RISK_LABELS:
                raise ValueError(f"nhãn không hợp lệ tại dòng {line_number}: {row['label']}")
            labels[event_id] = (label, row["annotator_note"].strip())
    return labels


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Truyền nhãn cuối của event sang snapshot cùng event_id."
    )
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        summary = propagate_event_labels(args.events, args.snapshots, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Không thể truyền nhãn: {exc}")
        return 1
    print(
        f"Đã ghi {args.output}: {summary['updated']} snapshot có nhãn, "
        f"{summary['unlabeled']} chưa gán nhãn, {summary['orphaned']} thiếu event."
    )
    return 1 if summary["orphaned"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
