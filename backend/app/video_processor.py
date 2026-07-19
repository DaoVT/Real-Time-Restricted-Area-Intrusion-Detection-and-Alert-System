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

from .image_filters import process_frame
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
        # Lấy điểm giữa cạnh dưới của khung người, tạm coi đó là vị trí chân.
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


class PersonDetector:
    def __init__(self, confidence: float = 0.35, model_name: str = "best.pt") -> None:
        self.confidence = confidence
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics is not installed or cannot be imported") from exc
        self.model = YOLO(model_name)
        self.device = select_yolo_device()

    def track(self, frame: np.ndarray) -> list[Detection]:
        # YOLO tìm người trong frame, ByteTrack gán ID để biết người này qua nhiều frame.
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.confidence,
            device=self.device,
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


def scale_detections(detections: list[Detection], scale_x: float, scale_y: float) -> list[Detection]:
    # Nếu đã thu nhỏ ảnh để chạy nhanh hơn, cần đổi khung người về kích thước video gốc.
    if scale_x == 1.0 and scale_y == 1.0:
        return detections
    scaled: list[Detection] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        scaled.append(
            Detection(
                track_id=detection.track_id,
                bbox=(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                confidence=detection.confidence,
            )
        )
    return scaled


class ZoneDetector:
    def __init__(self, normalized_points: Iterable[Point | dict], width: int, height: int) -> None:
        # UI gửi điểm từ 0 đến 1, backend nhân với kích thước video để ra tọa độ pixel.
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

    def draw(self, frame: np.ndarray, is_active: bool = False) -> None:
        color = (0, 0, 255) if is_active else (0, 180, 255)
        # Vẽ viền ngoài
        cv2.polylines(frame, [self.points], isClosed=True, color=color, thickness=2)
        # Vẽ phần tô màu bán trong suốt bên trong
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.points], color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)


class IntrusionEventDetector:
    def __init__(self, confirm_frames: int) -> None:
        self.confirm_frames = confirm_frames
        self._inside_counts: dict[int, int] = {}
        self._alerted_tracks: set[int] = set()

    def update(self, detections: list[Detection], zone: ZoneDetector) -> tuple[list[Detection], int]:
        active_tracks = {detection.track_id for detection in detections}
        
        # 1. Xử lý các track_id bị mất dấu (tracker mất do che khuất hoặc nhiễu)
        for track_id in list(self._inside_counts):
            if track_id not in active_tracks:
                # Giảm dần bộ đếm (debounce) thay vì xóa ngay lập tức
                current = self._inside_counts[track_id]
                if current > 0:
                    self._inside_counts[track_id] = current - 1
                else:
                    self._inside_counts.pop(track_id, None)
                    self._alerted_tracks.discard(track_id)

        inside_count = 0
        new_intrusions: list[Detection] = []
        
        # 2. Cập nhật trạng thái cho các track_id đang hiển thị
        for detection in detections:
            if zone.contains(detection.bottom_midpoint):
                inside_count += 1
                count = self._inside_counts.get(detection.track_id, 0) + 1
                # Giới hạn count tối đa bằng confirm_frames + 5 để tránh số đếm tăng vô tận
                self._inside_counts[detection.track_id] = min(count, self.confirm_frames + 5)
                
                # Chỉ báo động khi đủ số frame liên tục và chưa báo động trước đó
                if count >= self.confirm_frames and detection.track_id not in self._alerted_tracks:
                    self._alerted_tracks.add(detection.track_id)
                    new_intrusions.append(detection)
            else:
                # Nếu bước ra khỏi vùng, giảm count dần dần (debounce)
                current = self._inside_counts.get(detection.track_id, 0)
                if current > 0:
                    self._inside_counts[detection.track_id] = current - 1
                elif detection.track_id in self._alerted_tracks:
                    # Khi count thực sự về 0 (đã ra ngoài hoàn toàn), mới gỡ trạng thái báo động
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
        process_width: int | None = None,
        frame_stride: int = 1,
        save_output_video: bool = True,
        filter_type: str = "none",
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
        self.process_width = process_width
        self.frame_stride = max(frame_stride, 1)
        self.save_output_video = save_output_video
        self.filter_type = filter_type

    def run(self) -> dict:
        # Đây là luồng chính: mở video, đọc từng frame, phát hiện người và ghi kết quả.
        # Mỗi job có một thư mục output riêng để lưu CSV, ảnh bằng chứng và video kết quả.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir = self.output_dir / "evidence"
        evidence_dir.mkdir(exist_ok=True)
        events_csv = self.output_dir / "events.csv"
        output_video = self.output_dir / "result.mp4"

        # OpenCV mở file video đầu vào. Sau đó vòng lặp bên dưới sẽ đọc lần lượt từng frame.
        capture = cv2.VideoCapture(str(self.input_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {self.input_path}")
        writer = None
        try:
            # Lấy thông tin cơ bản của video để tính tiến độ, FPS và tạo video output đúng kích thước.
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            output_fps = max(source_fps / self.frame_stride, 1.0)
            if self.save_output_video:
                # VideoWriter dùng để ghi lại video đã vẽ khung người, điểm chân và vùng cấm.
                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("cannot create output video writer")

            # detector tìm người; zone đổi vùng cấm từ tọa độ 0..1 sang pixel của video.
            detector = self.detector or PersonDetector(self.confidence)
            zone = ZoneDetector(self.zone_points, width, height)
            # intrusion_detector nhớ trạng thái từng người để biết khi nào đủ điều kiện báo động.
            intrusion_detector = IntrusionEventDetector(self.confirm_frames)
            intrusion_count = 0
            frame_index = 0
            processed_frame_count = 0
            # Chỉ gửi một phần frame lên UI để preview nhẹ hơn, tối đa khoảng 10 FPS.
            preview_interval = max(1, int(output_fps / 10))
            # Gửi dữ liệu thống kê lên UI khoảng 5 lần/giây để tránh làm giật giao diện web
            stat_interval = max(1, int(output_fps / 5))
            started_at = time.perf_counter()

            with events_csv.open("w", newline="", encoding="utf-8") as csv_file:
                # CSV lưu danh sách các lần xâm nhập đã được xác nhận.
                csv_writer = csv.DictWriter(csv_file, fieldnames=["frame", "track_id", "confidence", "evidence_path"])
                csv_writer.writeheader()
                while not self.stop_event.is_set():
                    # Đọc frame tiếp theo. Nếu hết video thì thoát vòng lặp.
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_index += 1
                    processed_frame_count += 1
                    # --- ĐÁNH GIÁ AUTO FILTER 1 LẦN DUY NHẤT ---
                    if self.filter_type == "auto":
                        from .image_filters import detect_best_filter
                        self.filter_type = detect_best_filter(frame)
                        print(f"Auto-detected image filter: {self.filter_type}")

                    # --- ÁP DỤNG TIỀN XỬ LÝ ẢNH TỪ FILE image_filters.py ---
                    frame = process_frame(frame, filter_type=self.filter_type)
                    # -------------------------------------------------------

                    # Fast Mode có thể thu nhỏ frame trước khi đưa vào YOLO để tăng tốc.
                    detect_frame, scale_x, scale_y = prepare_detection_frame(frame, self.process_width)

                    # YOLO + ByteTrack trả về danh sách người, gồm ID, khung người và độ tin cậy.
                    detections = scale_detections(detector.track(detect_frame), scale_x, scale_y)
                    # Kiểm tra từng người có đứng trong vùng cấm đủ lâu để tạo cảnh báo không.
                    intrusions, inside_count = intrusion_detector.update(detections, zone)

                    # Vẽ kết quả lên frame: khung người màu xanh và điểm chân màu đỏ.
                    for detection in detections:
                        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 160, 35), 2)
                        cv2.circle(frame, tuple(int(v) for v in detection.bottom_midpoint), 4, (0, 0, 255), -1)
                    # Vẽ polygon vùng cấm lên frame (chuyển sang đỏ nếu có người bên trong).
                    zone.draw(frame, is_active=(inside_count > 0))

                    intrusion_events = []
                    for detection in intrusions:
                        # Khi có cảnh báo mới, lưu ảnh bằng chứng và ghi một dòng vào CSV.
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

                    if writer is not None:
                        # Nếu bật lưu video output, ghi frame đã vẽ overlay vào result.mp4.
                        writer.write(frame)
                    elapsed = max(time.perf_counter() - started_at, 0.001)
                    # Thống kê này được JobManager gửi lên frontend qua WebSocket.
                    stats = {
                        "progress": min(frame_index / total_frames, 1.0),
                        "fps": frame_index / elapsed,
                        "people_count": len(detections),
                        "inside_count": inside_count,
                        "intrusion_count": intrusion_count,
                        "events": intrusion_events,
                    }
                    if self.progress_callback and processed_frame_count % stat_interval == 0:
                        self.progress_callback(stats)
                    if self.frame_callback and processed_frame_count % preview_interval == 0:
                        # Gửi preview frame đã nén JPEG để UI hiển thị video realtime.
                        self.frame_callback(encode_preview_frame(frame))
                    for _ in range(self.frame_stride - 1):
                        # Fast Mode bỏ qua một số frame để chạy nhanh hơn.
                        if self.stop_event.is_set():
                            break
                        if not capture.grab():
                            break
                        frame_index += 1

            return {
                # Kết quả cuối cùng được JobManager lưu vào snapshot của job.
                "stopped": self.stop_event.is_set(),
                "output_video": str(output_video) if self.save_output_video else None,
                "events_csv": str(events_csv),
                "intrusion_count": intrusion_count,
            }
        finally:
            # Luôn giải phóng file video, kể cả khi job lỗi hoặc bị stop giữa chừng.
            capture.release()
            if writer is not None:
                writer.release()


def encode_preview_frame(frame: np.ndarray) -> bytes:
    # Tăng giới hạn độ phân giải preview gửi lên web từ 360p lên HD 720p
    height, width = frame.shape[:2]
    scale = min(1280 / width, 720 / height, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    # Tăng chất lượng nén JPEG từ 80 lên 95 để video trên web mượt và nét hơn
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("failed to encode preview frame")
    return encoded.tobytes()


def prepare_detection_frame(frame: np.ndarray, process_width: int | None) -> tuple[np.ndarray, float, float]:
    # Thu nhỏ frame trước khi đưa vào YOLO để tăng tốc, nếu người dùng bật Fast Mode.
    if process_width is None:
        return frame, 1.0, 1.0
    height, width = frame.shape[:2]
    if width <= process_width:
        return frame, 1.0, 1.0
    scale = process_width / width
    resized_width = process_width
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height))
    return resized, width / resized_width, height / resized_height


def select_yolo_device() -> int | str:
    # Có GPU thì dùng GPU cho nhanh, không có thì chạy bằng CPU.
    try:
        import torch
    except Exception:
        return "cpu"
    return 0 if torch.cuda.is_available() else "cpu"


def load_zone_file(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = data.get("points", data.get("zone_points"))
    if not isinstance(data, list):
        raise ValueError("zone file must contain a list of normalized points")
    return data
