from __future__ import annotations

import csv

import numpy as np
import pytest

from backend.app.risk import DatasetExporter, FeatureExtractor
from backend.app.video_processor import Detection, VideoProcessor
from high_level_training.data.prepare_dataset import combine_csv_files


def test_feature_extractor_calculates_motion_and_periodic_snapshot():
    extractor = FeatureExtractor(snapshot_interval_seconds=0.5)

    # Giai đoạn CONFIRMING được tích lũy nhưng chưa có event_id nên chưa xuất snapshot.
    assert extractor.observe(
        track_id=7,
        event_id=None,
        entry_time_sec=0.0,
        video_time_sec=0.0,
        foot_point=(0.0, 0.0),
        frame_diagonal=100.0,
        zone_depth_pixels=10.0,
        people_in_zone=1,
        reentry_count=0,
        detection_confidence=0.8,
    ) is None

    # FSM xác nhận INTRUDING tại 0,1 giây; snapshot đầu tiên chỉ xuất sau thêm 0,5 giây.
    assert extractor.observe(
        track_id=7,
        event_id="event-7",
        entry_time_sec=0.0,
        video_time_sec=0.1,
        foot_point=(3.0, 4.0),
        frame_diagonal=100.0,
        zone_depth_pixels=20.0,
        people_in_zone=2,
        reentry_count=0,
        detection_confidence=0.9,
    ) is None

    snapshot = extractor.observe(
        track_id=7,
        event_id="event-7",
        entry_time_sec=0.0,
        video_time_sec=0.6,
        foot_point=(3.0, 4.0),
        frame_diagonal=100.0,
        zone_depth_pixels=15.0,
        people_in_zone=1,
        reentry_count=1,
        detection_confidence=0.7,
    )

    assert snapshot is not None
    assert snapshot.inside_duration_sec == pytest.approx(0.6)
    assert snapshot.average_speed_norm == pytest.approx(0.05 / 0.6)
    assert snapshot.max_speed_norm == pytest.approx(0.5)
    assert snapshot.max_zone_depth_norm == pytest.approx(0.2)
    assert snapshot.stopped_duration_sec == pytest.approx(0.5)
    assert snapshot.reentry_count == 1
    assert snapshot.people_in_zone_peak == 2
    assert snapshot.detection_confidence_avg == pytest.approx(0.8)


def test_mark_outside_prevents_speed_jump_across_reentry():
    extractor = FeatureExtractor(snapshot_interval_seconds=0.5)
    extractor.observe(
        track_id=1,
        event_id="event-1",
        entry_time_sec=0.0,
        video_time_sec=0.0,
        foot_point=(0.0, 0.0),
        frame_diagonal=100.0,
        zone_depth_pixels=5.0,
        people_in_zone=1,
        reentry_count=0,
        detection_confidence=0.9,
    )
    extractor.mark_outside(1)
    snapshot = extractor.observe(
        track_id=1,
        event_id="event-1",
        entry_time_sec=0.0,
        video_time_sec=0.5,
        foot_point=(90.0, 90.0),
        frame_diagonal=100.0,
        zone_depth_pixels=5.0,
        people_in_zone=1,
        reentry_count=1,
        detection_confidence=0.9,
    )

    assert snapshot is not None
    assert snapshot.average_speed_norm == 0.0
    assert snapshot.max_speed_norm == 0.0


def test_dataset_exporter_writes_event_and_snapshot_with_empty_label(tmp_path):
    extractor = FeatureExtractor(snapshot_interval_seconds=0.5)
    extractor.observe(
        track_id=2,
        event_id="event-2",
        entry_time_sec=0.0,
        video_time_sec=0.0,
        foot_point=(10.0, 10.0),
        frame_diagonal=100.0,
        zone_depth_pixels=5.0,
        people_in_zone=1,
        reentry_count=0,
        detection_confidence=0.8,
    )
    snapshot = extractor.observe(
        track_id=2,
        event_id="event-2",
        entry_time_sec=0.0,
        video_time_sec=0.5,
        foot_point=(11.0, 10.0),
        frame_diagonal=100.0,
        zone_depth_pixels=6.0,
        people_in_zone=1,
        reentry_count=0,
        detection_confidence=0.8,
    )
    assert snapshot is not None
    final_vector = extractor.complete(2, "event-2", 0.7)
    assert final_vector is not None

    with DatasetExporter(tmp_path) as exporter:
        exporter.write_snapshot("video-a", snapshot)
        exporter.write_event("video-a", final_vector, camera_id="WEB_CAM_001", evidence_path="evidence/event-2.jpg")

    with (tmp_path / "high_level_snapshots.csv").open(encoding="utf-8") as file:
        [snapshot_row] = list(csv.DictReader(file))
    with (tmp_path / "high_level_events.csv").open(encoding="utf-8") as file:
        [event_row] = list(csv.DictReader(file))

    assert snapshot_row["event_id"] == "event-2"
    assert snapshot_row["label"] == ""
    assert event_row["video_id"] == "video-a"
    assert event_row["camera_id"] == "WEB_CAM_001"
    assert event_row["evidence_path"] == "evidence/event-2.jpg"
    assert event_row["event_duration_sec"] == "0.7"


def test_prepare_dataset_combines_job_csv_files(tmp_path):
    source = tmp_path / "outputs"
    for job_name, event_id in (("job-a", "event-a"), ("job-b", "event-b")):
        extractor = FeatureExtractor(snapshot_interval_seconds=0.5)
        extractor.observe(
            track_id=1,
            event_id=event_id,
            entry_time_sec=0.0,
            video_time_sec=0.0,
            foot_point=(0.0, 0.0),
            frame_diagonal=100.0,
            zone_depth_pixels=1.0,
            people_in_zone=1,
            reentry_count=0,
            detection_confidence=0.9,
        )
        vector = extractor.complete(1, event_id, 1.0)
        assert vector is not None
        with DatasetExporter(source / job_name) as exporter:
            exporter.write_event(job_name, vector)

    destination = tmp_path / "dataset" / "high_level_events.csv"
    count = combine_csv_files(source, "high_level_events.csv", destination)

    with destination.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert count == 2
    assert {row["event_id"] for row in rows} == {"event-a", "event-b"}


def test_video_processor_creates_high_level_dataset_files(tmp_path, monkeypatch):
    class FakeCapture:
        def __init__(self) -> None:
            self.frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(9)]
            self.index = 0
            self.released = False

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            values = {3: 100.0, 4: 100.0, 5: 10.0, 7: 9.0}
            return values.get(property_id, 0.0)

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame.copy()

        def grab(self) -> bool:
            self.index += 1
            return self.index <= len(self.frames)

        def release(self) -> None:
            self.released = True

    class FakeDetector:
        def track(self, frame: np.ndarray) -> list[Detection]:
            return [Detection(track_id=1, bbox=(20, 20, 40, 40), confidence=0.9)]

    class FakeRiskPredictor:
        def predict(self, snapshot):
            return {
                "available": True,
                "event_id": snapshot.event_id,
                "track_id": snapshot.track_id,
                "video_time_sec": snapshot.sample_time_sec,
                "level": "MEDIUM",
                "probabilities": {"LOW": 0.1, "MEDIUM": 0.8, "HIGH": 0.1},
            }

    capture = FakeCapture()
    risk_messages = []
    progress_messages = []
    monkeypatch.setattr("backend.app.video_processor.cv2.VideoCapture", lambda _: capture)
    result = VideoProcessor(
        input_path=tmp_path / "video.mp4",
        output_dir=tmp_path / "output",
        zone_points=[
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 0.0},
            {"x": 1.0, "y": 1.0},
            {"x": 0.0, "y": 1.0},
        ],
        confidence=0.35,
        # Sự kiện xuất hiện ở frame xử lý thứ 3, không trùng nhịp thống kê mỗi 2 frame.
        confirm_frames=3,
        detector=FakeDetector(),
        save_output_video=False,
        camera_id="TEST_CAM_001",
        progress_callback=progress_messages.append,
        risk_predictor=FakeRiskPredictor(),
        risk_callback=risk_messages.append,
    ).run()

    assert capture.released is True
    assert result["high_level_events_csv"] == str(tmp_path / "output" / "high_level_events.csv")
    with (tmp_path / "output" / "high_level_events.csv").open(encoding="utf-8") as file:
        event_rows = list(csv.DictReader(file))
    with (tmp_path / "output" / "high_level_snapshots.csv").open(encoding="utf-8") as file:
        snapshot_rows = list(csv.DictReader(file))
    assert len(event_rows) == 1
    assert len(snapshot_rows) == 1
    assert event_rows[0]["camera_id"] == "TEST_CAM_001"
    assert event_rows[0]["evidence_path"]
    assert event_rows[0]["label"] == ""
    assert len(risk_messages) == 1
    assert risk_messages[0]["level"] == "MEDIUM"
    assert risk_messages[0]["track_id"] == 1
    delivered_events = [
        event
        for message in progress_messages
        for event in message.get("events", [])
    ]
    assert len(delivered_events) == 1
    assert delivered_events[0]["track_id"] == 1
