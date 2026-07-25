"""Gộp kết quả của nhiều video thành một dataset chung.

Mỗi job có một file event và một file snapshot.
Công cụ tìm tất cả file cùng loại, kiểm tra tên cột rồi nối các dòng lại.
Nhãn trống vẫn được giữ để gán nhãn ở bước tiếp theo.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


DATASET_FILES = ("high_level_events.csv", "high_level_snapshots.csv")


def combine_csv_files(source_root: Path, filename: str, destination_path: Path) -> int:
    """Nối các CSV cùng loại và trả về số dòng đã ghi vào file tổng hợp."""

    source_root = Path(source_root)
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # BƯỚC 1: Tìm mọi file CSV cùng tên trong các thư mục output.
    source_files = [
        path
        for path in sorted(source_root.rglob(filename))
        if path.resolve() != destination_path.resolve()
    ]

    fieldnames: list[str] | None = None
    row_count = 0
    # BƯỚC 2: Lần lượt chép dữ liệu, nhưng chỉ khi các file có cùng danh sách cột.
    with destination_path.open("w", newline="", encoding="utf-8") as output_file:
        writer: csv.DictWriter | None = None
        for source_path in source_files:
            with source_path.open("r", newline="", encoding="utf-8-sig") as input_file:
                reader = csv.DictReader(input_file)
                current_fields = reader.fieldnames
                if not current_fields:
                    continue
                if fieldnames is None:
                    fieldnames = current_fields
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    writer.writeheader()
                elif current_fields != fieldnames:
                    raise ValueError(f"CSV schema does not match: {source_path}")

                # Chỉ gộp dữ liệu; nhãn trống vẫn được giữ để người dùng gán nhãn sau.
                for row in reader:
                    if writer is not None:
                        writer.writerow(row)
                        row_count += 1
    return row_count


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Gộp dataset mức cao từ nhiều thư mục output của job.")
    parser.add_argument("--source", type=Path, default=Path("backend/output"), help="Thư mục chứa output các job.")
    parser.add_argument("--destination", type=Path, required=True, help="Thư mục dataset tổng hợp, nên đặt ở ổ D.")
    args = parser.parse_args()

    for filename in DATASET_FILES:
        count = combine_csv_files(args.source, filename, args.destination / filename)
        print(f"{filename}: {count} dòng")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
