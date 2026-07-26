"""Lưu dữ liệu hành vi mức cao thành CSV.

File snapshot ghi trạng thái đang diễn ra của sự kiện sau mỗi 0,5 giây.
File event ghi kết quả tổng hợp một lần khi sự kiện kết thúc.
Hai file này là dữ liệu đầu vào cho bước gán nhãn và train Random Forest.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

from .feature_extractor import FeatureVector


FEATURE_COLUMNS = [
    "inside_duration_sec",
    "average_speed_norm",
    "max_speed_norm",
    "max_zone_depth_norm",
    "stopped_duration_sec",
    "reentry_count",
    "direction_change_count",
    "people_in_zone_peak",
    "observation_count",
    # Confidence được lưu để phân tích chất lượng phát hiện, chưa nên dùng làm đầu vào model rủi ro.
    "detection_confidence_avg",
]

SNAPSHOT_COLUMNS = [
    "event_id",
    "video_id",
    "camera_id",
    "track_id",
    "video_time_sec",
    "elapsed_sec",
    *FEATURE_COLUMNS,
    "label",
    "annotator_note",
]

EVENT_COLUMNS = [
    "event_id",
    "video_id",
    "camera_id",
    "track_id",
    "entry_time_sec",
    "confirmed_time_sec",
    "end_time_sec",
    "event_duration_sec",
    "evidence_path",
    *FEATURE_COLUMNS,
    "label",
    "annotator_note",
]


class DatasetExporter:
    """Ghi đặc trưng mức cao thành hai file CSV mà không phụ thuộc FastAPI."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.events_path = self.output_dir / "high_level_events.csv"
        self.snapshots_path = self.output_dir / "high_level_snapshots.csv"
        self._events_file: TextIO | None = None
        self._snapshots_file: TextIO | None = None
        self._events_writer: csv.DictWriter | None = None
        self._snapshots_writer: csv.DictWriter | None = None

    def __enter__(self) -> DatasetExporter:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._events_file = self.events_path.open("w", newline="", encoding="utf-8")
        self._snapshots_file = self.snapshots_path.open("w", newline="", encoding="utf-8")
        self._events_writer = csv.DictWriter(self._events_file, fieldnames=EVENT_COLUMNS)
        self._snapshots_writer = csv.DictWriter(self._snapshots_file, fieldnames=SNAPSHOT_COLUMNS)
        self._events_writer.writeheader()
        self._snapshots_writer.writeheader()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        # Đóng file ngay cả khi xử lý video lỗi hoặc bị dừng giữa chừng.
        if self._events_file is not None:
            self._events_file.close()
        if self._snapshots_file is not None:
            self._snapshots_file.close()

    def write_snapshot(
        self,
        video_id: str,
        vector: FeatureVector,
        camera_id: str = "",
        label: str = "",
        annotator_note: str = "",
    ) -> None:
        """Ghi đặc trưng giữa sự kiện để sau này train dự đoán rủi ro theo thời gian thực."""

        if self._snapshots_writer is None or self._snapshots_file is None:
            raise RuntimeError("DatasetExporter must be used as a context manager")
        row = self._common_row(vector)
        row.update(
            {
                "video_id": video_id,
                "camera_id": camera_id,
                "video_time_sec": vector.sample_time_sec,
                "elapsed_sec": vector.elapsed_sec,
                "label": label,
                "annotator_note": annotator_note,
            }
        )
        self._snapshots_writer.writerow(row)
        self._snapshots_file.flush()

    def write_event(
        self,
        video_id: str,
        vector: FeatureVector,
        camera_id: str = "",
        evidence_path: str = "",
        label: str = "",
        annotator_note: str = "",
    ) -> None:
        """Ghi một dòng cuối cùng sau khi FSM xác nhận người đã rời vùng hoặc hết video."""

        if self._events_writer is None or self._events_file is None:
            raise RuntimeError("DatasetExporter must be used as a context manager")
        row = self._common_row(vector)
        row.update(
            {
                "video_id": video_id,
                "camera_id": camera_id,
                "entry_time_sec": vector.entry_time_sec,
                "confirmed_time_sec": vector.confirmed_time_sec,
                "end_time_sec": vector.sample_time_sec,
                "event_duration_sec": vector.elapsed_sec,
                "evidence_path": evidence_path,
                "label": label,
                "annotator_note": annotator_note,
            }
        )
        self._events_writer.writerow(row)
        self._events_file.flush()

    @staticmethod
    def _common_row(vector: FeatureVector) -> dict[str, object]:
        return {
            "event_id": vector.event_id,
            "track_id": vector.track_id,
            "inside_duration_sec": vector.inside_duration_sec,
            "average_speed_norm": vector.average_speed_norm,
            "max_speed_norm": vector.max_speed_norm,
            "max_zone_depth_norm": vector.max_zone_depth_norm,
            "stopped_duration_sec": vector.stopped_duration_sec,
            "reentry_count": vector.reentry_count,
            "direction_change_count": vector.direction_change_count,
            "people_in_zone_peak": vector.people_in_zone_peak,
            "observation_count": vector.observation_count,
            "detection_confidence_avg": vector.detection_confidence_avg,
        }
