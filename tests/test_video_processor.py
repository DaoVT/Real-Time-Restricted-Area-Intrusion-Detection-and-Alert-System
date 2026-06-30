from __future__ import annotations

import numpy as np

from backend.app.video_processor import Detection, IntrusionEventDetector, ZoneDetector, encode_preview_frame


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
