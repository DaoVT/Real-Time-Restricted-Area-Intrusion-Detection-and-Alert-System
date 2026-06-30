from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Iterable
from uuid import uuid4

import cv2
import numpy as np

from .schemas import Point


ProgressCallback = Callable[[dict], None]
FrameCallback = Callable[[bytes], None]


@dataclass(frozen=True)
class Detection:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def bottom_midpoint(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


class PersonDetector:
    def __init__(self, confidence: float = 0.35, model_name: str = "yolov8n.pt") -> None:
        self.confidence = confidence
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics is not installed or cannot be imported") from exc
        self.model = YOLO(model_name)

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.confidence,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return detections
        for box, track_id, confidence in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy(), boxes.conf.cpu().numpy()):
            detections.append(Detection(int(track_id), tuple(float(v) for v in box), float(confidence)))
        return detections


class ZoneDetector:
    def __init__(self, normalized_points: Iterable[Point | dict], width: int, height: int) -> None:
        self.points = np.array(
            [
                (
                    int((point.x if isinstance(point, Point) else point["x"]) * width),
                    int((point.y if isinstance(point, Point) else point["y"]) * height),
                )
                for point in normalized_points
            ],
            dtype=np.int32,
        )

    def contains(self, point: tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.points, point, False) >= 0

    def draw(self, frame: np.ndarray) -> None:
        cv2.polylines(frame, [self.points], isClosed=True, color=(0, 180, 255), thickness=2)


class IntrusionEventDetector:
    def __init__(self, confirm_frames: int) -> None:
        self.confirm_frames = confirm_frames
        self._inside_counts: dict[int, int] = {}
        self._alerted_tracks: set[int] = set()

    def update(self, detections: list[Detection], zone: ZoneDetector) -> tuple[list[Detection], int]:
        active_tracks = {detection.track_id for detection in detections}
        for track_id in list(self._inside_counts):
            if track_id not in active_tracks:
                self._inside_counts.pop(track_id, None)
                self._alerted_tracks.discard(track_id)

        inside_count = 0
        new_intrusions: list[Detection] = []
        for detection in detections:
            if zone.contains(detection.bottom_midpoint):
                inside_count += 1
                count = self._inside_counts.get(detection.track_id, 0) + 1
                self._inside_counts[detection.track_id] = count
                if count >= self.confirm_frames and detection.track_id not in self._alerted_tracks:
                    self._alerted_tracks.add(detection.track_id)
                    new_intrusions.append(detection)
            else:
                self._inside_counts[detection.track_id] = 0
                self._alerted_tracks.discard(detection.track_id)
        return new_intrusions, inside_count


class VideoProcessor:
    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        zone_points: list[Point | dict],
        confidence: float,
        confirm_frames: int,
        progress_callback: ProgressCallback | None = None,
        frame_callback: FrameCallback | None = None,
        stop_event: Event | None = None,
        detector: PersonDetector | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.zone_points = zone_points
        self.confidence = confidence
        self.confirm_frames = confirm_frames
        self.progress_callback = progress_callback
        self.frame_callback = frame_callback
        self.stop_event = stop_event or Event()
        self.detector = detector

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir = self.output_dir / "evidence"
        evidence_dir.mkdir(exist_ok=True)
        events_csv = self.output_dir / "events.csv"
        output_video = self.output_dir / "result.mp4"

        capture = cv2.VideoCapture(str(self.input_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {self.input_path}")
        writer = None
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            writer = cv2.VideoWriter(
                str(output_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                source_fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("cannot create output video writer")

            detector = self.detector or PersonDetector(self.confidence)
            zone = ZoneDetector(self.zone_points, width, height)
            intrusion_detector = IntrusionEventDetector(self.confirm_frames)
            intrusion_count = 0
            frame_index = 0
            preview_interval = max(1, int(source_fps / 10))
            started_at = time.perf_counter()

            with events_csv.open("w", newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=["frame", "track_id", "confidence", "evidence_path"])
                csv_writer.writeheader()
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_index += 1
                    detections = detector.track(frame)
                    intrusions, inside_count = intrusion_detector.update(detections, zone)

                    for detection in detections:
                        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 160, 35), 2)
                        cv2.circle(frame, tuple(int(v) for v in detection.bottom_midpoint), 4, (0, 0, 255), -1)
                    zone.draw(frame)

                    intrusion_events = []
                    for detection in intrusions:
                        intrusion_count += 1
                        evidence_path = evidence_dir / f"{uuid4()}.jpg"
                        cv2.imwrite(str(evidence_path), frame)
                        event = {
                            "frame": frame_index,
                            "track_id": detection.track_id,
                            "confidence": detection.confidence,
                            "evidence_path": str(evidence_path),
                        }
                        csv_writer.writerow(event)
                        intrusion_events.append(event)

                    writer.write(frame)
                    elapsed = max(time.perf_counter() - started_at, 0.001)
                    stats = {
                        "progress": min(frame_index / total_frames, 1.0),
                        "fps": frame_index / elapsed,
                        "people_count": len(detections),
                        "inside_count": inside_count,
                        "intrusion_count": intrusion_count,
                        "events": intrusion_events,
                    }
                    if self.progress_callback:
                        self.progress_callback(stats)
                    if self.frame_callback and frame_index % preview_interval == 0:
                        self.frame_callback(encode_preview_frame(frame))

            return {
                "stopped": self.stop_event.is_set(),
                "output_video": str(output_video),
                "events_csv": str(events_csv),
                "intrusion_count": intrusion_count,
            }
        finally:
            capture.release()
            if writer is not None:
                writer.release()


def encode_preview_frame(frame: np.ndarray) -> bytes:
    height, width = frame.shape[:2]
    scale = min(640 / width, 360 / height, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("failed to encode preview frame")
    return encoded.tobytes()


def load_zone_file(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = data.get("points", data.get("zone_points"))
    if not isinstance(data, list):
        raise ValueError("zone file must contain a list of normalized points")
    return data
