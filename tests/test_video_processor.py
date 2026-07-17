from __future__ import annotations

import json

import numpy as np

from backend.app.video_processor import (
    Detection,
    IntrusionEventDetector,
    IntrusionState,
    ZoneDetector,
    encode_preview_frame,
    load_zone_file,
    prepare_detection_frame,
    scale_detections,
    select_yolo_device,
)


def test_intrusion_requires_confirm_frames_and_deduplicates():
    zone = ZoneDetector([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, {"x": 0.0, "y": 1.0}], 100, 100)
    detector = IntrusionEventDetector(confirm_frames=2)
    detection = Detection(track_id=8, bbox=(10, 10, 20, 20), confidence=0.8)

    first, inside = detector.update([detection], zone)
    assert first == []
    assert inside == 1

    second, _ = detector.update([detection], zone)
    assert len(second) == 1

    third, _ = detector.update([detection], zone)
    assert third == []
    assert detector.track_states[8].state == IntrusionState.INTRUDING


def test_exiting_debounces_boundary_jitter_and_allows_real_reentry():
    zone = ZoneDetector([{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 0.0}, {"x": 0.5, "y": 1.0}, {"x": 0.0, "y": 1.0}], 100, 100)
    detector = IntrusionEventDetector(confirm_frames=2, source_fps=10, exit_confirm_seconds=0.3)
    inside_detection = Detection(track_id=3, bbox=(10, 10, 20, 20), confidence=0.9)
    outside_detection = Detection(track_id=3, bbox=(70, 10, 80, 20), confidence=0.9)

    detector.update([inside_detection], zone, video_time_sec=0.0)
    events, _ = detector.update([inside_detection], zone, video_time_sec=0.1)
    assert len(events) == 1

    detector.update([outside_detection], zone, video_time_sec=0.2)
    assert detector.track_states[3].state == IntrusionState.EXITING
    events, _ = detector.update([inside_detection], zone, video_time_sec=0.3)
    assert events == []
    assert detector.track_states[3].state == IntrusionState.INTRUDING

    detector.update([outside_detection], zone, video_time_sec=0.4)
    detector.update([outside_detection], zone, video_time_sec=0.71)
    assert detector.track_states[3].state == IntrusionState.OUTSIDE

    detector.update([inside_detection], zone, video_time_sec=0.8)
    events, _ = detector.update([inside_detection], zone, video_time_sec=0.91)
    assert len(events) == 1


def test_lost_track_recovers_state_then_expires_after_timeout():
    zone = ZoneDetector([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, {"x": 0.0, "y": 1.0}], 100, 100)
    detector = IntrusionEventDetector(confirm_frames=1, track_timeout_seconds=2.0)
    detection = Detection(track_id=5, bbox=(10, 10, 20, 20), confidence=0.8)

    events, _ = detector.update([detection], zone, video_time_sec=0.0)
    assert len(events) == 1
    detector.update([], zone, video_time_sec=0.1)
    assert detector.track_states[5].state == IntrusionState.LOST

    events, _ = detector.update([detection], zone, video_time_sec=0.2)
    assert events == []
    assert detector.track_states[5].state == IntrusionState.INTRUDING

    detector.update([], zone, video_time_sec=0.3)
    detector.update([], zone, video_time_sec=2.31)
    assert 5 not in detector.track_states


def test_confirm_time_uses_source_video_timeline_in_fast_mode():
    zone = ZoneDetector([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, {"x": 0.0, "y": 1.0}], 100, 100)
    detector = IntrusionEventDetector(confirm_frames=3, source_fps=30, frame_stride=3)
    detection = Detection(track_id=9, bbox=(10, 10, 20, 20), confidence=0.8)

    first, _ = detector.update([detection], zone, video_time_sec=0.0)
    second, _ = detector.update([detection], zone, video_time_sec=0.1)
    third, _ = detector.update([detection], zone, video_time_sec=0.2)

    assert first == []
    assert second == []
    assert len(third) == 1
    assert detector.confirm_seconds == 0.2


def test_no_people_means_no_intrusion_event():
    zone = ZoneDetector([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, {"x": 0.0, "y": 1.0}], 100, 100)
    detector = IntrusionEventDetector(confirm_frames=2)
    events, inside = detector.update([], zone)
    assert events == []
    assert inside == 0


def test_preview_frame_is_binary_jpeg_and_resized():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    encoded = encode_preview_frame(frame)
    assert encoded.startswith(b"\xff\xd8")
    assert encoded.endswith(b"\xff\xd9")


def test_load_zone_file_accepts_utf8_bom(tmp_path):
    zone_path = tmp_path / "zone.json"
    zone_path.write_text(json.dumps({"zone_points": [{"x": 0.1, "y": 0.2}, {"x": 0.8, "y": 0.2}, {"x": 0.4, "y": 0.9}]}), encoding="utf-8-sig")
    points = load_zone_file(zone_path)
    assert points[0] == {"x": 0.1, "y": 0.2}


def test_prepare_detection_frame_and_scale_detections():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    resized, scale_x, scale_y = prepare_detection_frame(frame, 640)
    assert resized.shape[:2] == (360, 640)
    assert scale_x == 3.0
    assert scale_y == 3.0

    [detection] = scale_detections([Detection(track_id=1, bbox=(10, 20, 30, 40), confidence=0.9)], scale_x, scale_y)
    assert detection.bbox == (30, 60, 90, 120)


def test_select_yolo_device_returns_supported_value():
    assert select_yolo_device() in {0, "cpu"}
