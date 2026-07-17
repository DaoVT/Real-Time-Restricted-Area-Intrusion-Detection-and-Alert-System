from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from enum import Enum
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
    def __init__(self, confidence: float = 0.35, model_name: str = "yolov8n.pt") -> None:
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


class IntrusionState(str, Enum):
    # OUTSIDE: người đang ở ngoài vùng cấm.
    OUTSIDE = "OUTSIDE"
    # CONFIRMING: người vừa vào vùng, cần ở đủ lâu trước khi xác nhận xâm nhập.
    CONFIRMING = "CONFIRMING"
    # INTRUDING: đã xác nhận xâm nhập và đã phát cảnh báo cho lần vào vùng này.
    INTRUDING = "INTRUDING"
    # EXITING: người vừa ra khỏi vùng, chờ thêm để tránh bounding box rung ở biên.
    EXITING = "EXITING"
    # LOST: ByteTrack tạm thời không nhìn thấy track nhưng chưa hết thời gian chờ.
    LOST = "LOST"


@dataclass
class StateTransition:
    # Một bản ghi cho biết FSM đổi từ trạng thái nào sang trạng thái nào tại thời gian video nào.
    video_time_sec: float
    from_state: IntrusionState
    to_state: IntrusionState


@dataclass
class TrackIntrusionState:
    # Mỗi track_id có một đối tượng riêng để trạng thái của nhiều người không ảnh hưởng lẫn nhau.
    state: IntrusionState = IntrusionState.OUTSIDE
    # Giữ trạng thái trước để có thể phục hồi sau khi ByteTrack tạm mất dấu.
    previous_state: IntrusionState = IntrusionState.OUTSIDE

    # Các mốc thời gian đều thuộc timeline video, không dùng tốc độ chạy thực tế của máy.
    first_seen_time_sec: float = 0.0
    last_seen_time_sec: float = 0.0
    entry_time_sec: float | None = None
    confirmed_time_sec: float | None = None
    exit_start_time_sec: float | None = None
    lost_start_time_sec: float | None = None

    # Các thuộc tính này mô tả phiên vào vùng hiện tại và phục vụ mở rộng model mức cao sau này.
    inside_frames: int = 0
    outside_frames: int = 0
    alerted: bool = False
    entry_count: int = 0
    event_id: str | None = None
    last_foot_point: tuple[float, float] | None = None
    state_history: list[StateTransition] = field(default_factory=list)


class IntrusionEventDetector:
    def __init__(
        self,
        confirm_frames: int,
        source_fps: float = 25.0,
        frame_stride: int = 1,
        exit_confirm_seconds: float = 0.3,
        track_timeout_seconds: float = 2.0,
    ) -> None:
        # Chuẩn hóa cấu hình để tránh ngưỡng âm hoặc bằng 0 làm FSM chuyển sai trạng thái.
        self.confirm_frames = max(confirm_frames, 1)
        self.source_fps = max(source_fps, 1.0)
        self.frame_stride = max(frame_stride, 1)
        self.exit_confirm_seconds = max(exit_confirm_seconds, 0.0)
        self.track_timeout_seconds = max(track_timeout_seconds, 0.0)
        # Từ điển trung tâm: khóa là track_id, giá trị là FSM của người tương ứng.
        self.track_states: dict[int, TrackIntrusionState] = {}
        # Chỉ dùng trong unit test hoặc nơi gọi không truyền timestamp video.
        self._synthetic_time_sec = 0.0

    @property
    def confirm_seconds(self) -> float:
        # Giữ tương thích confirm_frames nhưng đo trên timeline video gốc, kể cả khi Fast Mode bỏ frame.
        return (self.confirm_frames - 1) * self.frame_stride / self.source_fps

    def update(
        self,
        detections: list[Detection],
        zone: ZoneDetector,
        video_time_sec: float | None = None,
    ) -> tuple[list[Detection], int]:
        # Bước 1: xác định thời gian video và đánh dấu các track không còn xuất hiện là LOST.
        current_time = self._resolve_video_time(video_time_sec)
        active_tracks = {detection.track_id for detection in detections}
        self._mark_missing_tracks(active_tracks, current_time)

        # Bước 2: cập nhật FSM độc lập cho từng người đang được ByteTrack nhìn thấy.
        inside_count = 0
        new_intrusions: list[Detection] = []
        for detection in detections:
            state = self._get_or_create_state(detection.track_id, current_time)
            state.last_seen_time_sec = current_time
            state.last_foot_point = detection.bottom_midpoint

            if state.state == IntrusionState.LOST:
                # Cùng track_id xuất hiện lại trước timeout: quay về trạng thái trước khi mất dấu.
                resume_state = state.previous_state
                state.lost_start_time_sec = None
                self._transition(state, resume_state, current_time)

            # Điểm giữa cạnh dưới bounding box được coi là vị trí chân để kiểm tra polygon.
            if zone.contains(detection.bottom_midpoint):
                inside_count += 1
                if self._update_inside(state, current_time):
                    new_intrusions.append(detection)
            else:
                self._update_outside(state, current_time)

        return new_intrusions, inside_count

    def _resolve_video_time(self, video_time_sec: float | None) -> float:
        if video_time_sec is not None:
            self._synthetic_time_sec = max(video_time_sec, self._synthetic_time_sec)
            return video_time_sec
        current_time = self._synthetic_time_sec
        self._synthetic_time_sec += self.frame_stride / self.source_fps
        return current_time

    def _get_or_create_state(self, track_id: int, current_time: float) -> TrackIntrusionState:
        state = self.track_states.get(track_id)
        if state is None:
            # Track mới luôn bắt đầu từ OUTSIDE rồi mới được kiểm tra vị trí ở bước tiếp theo.
            state = TrackIntrusionState(
                first_seen_time_sec=current_time,
                last_seen_time_sec=current_time,
            )
            self.track_states[track_id] = state
        return state

    def _mark_missing_tracks(self, active_tracks: set[int], current_time: float) -> None:
        for track_id, state in list(self.track_states.items()):
            if track_id in active_tracks:
                continue
            if state.state != IntrusionState.LOST:
                # Không xóa ngay vì người có thể chỉ bị che khuất trong vài frame.
                state.previous_state = state.state
                state.lost_start_time_sec = current_time
                self._transition(state, IntrusionState.LOST, current_time)
                continue
            lost_since = state.lost_start_time_sec if state.lost_start_time_sec is not None else current_time
            if current_time - lost_since >= self.track_timeout_seconds:
                # Hết timeout mới dọn state để tránh giữ dữ liệu track cũ trong bộ nhớ.
                del self.track_states[track_id]

    def _update_inside(self, state: TrackIntrusionState, current_time: float) -> bool:
        # Track đang ở trong polygon nên tăng thời gian hiện diện và hủy bộ đếm ngoài vùng.
        state.inside_frames += 1
        state.outside_frames = 0

        if state.state == IntrusionState.OUTSIDE:
            # Lần đầu bước vào chỉ chuyển sang CONFIRMING, chưa phát cảnh báo ngay.
            state.entry_time_sec = current_time
            state.entry_count += 1
            state.alerted = False
            state.event_id = None
            self._transition(state, IntrusionState.CONFIRMING, current_time)

        if state.state == IntrusionState.EXITING:
            # Quay lại trước ngưỡng xác nhận ra ngoài: vẫn là cùng một sự kiện xâm nhập.
            state.exit_start_time_sec = None
            self._transition(state, IntrusionState.INTRUDING, current_time)
            return False

        entry_time = state.entry_time_sec if state.entry_time_sec is not None else current_time
        if state.state == IntrusionState.CONFIRMING and current_time - entry_time >= self.confirm_seconds:
            # Chỉ nhánh này trả True, vì vậy mỗi phiên vào vùng chỉ tạo đúng một cảnh báo mới.
            state.confirmed_time_sec = current_time
            state.alerted = True
            state.event_id = str(uuid4())
            self._transition(state, IntrusionState.INTRUDING, current_time)
            return True
        return False

    def _update_outside(self, state: TrackIntrusionState, current_time: float) -> None:
        # Track đang ở ngoài polygon nên chuỗi frame xác nhận bên trong không còn liên tục.
        state.outside_frames += 1
        state.inside_frames = 0

        if state.state == IntrusionState.CONFIRMING:
            # Ra ngoài trước khi đủ thời gian: hủy lần vào vùng chưa được xác nhận.
            self._reset_to_outside(state, current_time)
            return
        if state.state == IntrusionState.INTRUDING:
            # Chưa kết thúc sự kiện ngay; chuyển EXITING để chống rung ở mép polygon.
            state.exit_start_time_sec = current_time
            self._transition(state, IntrusionState.EXITING, current_time)
            return
        if state.state == IntrusionState.EXITING:
            exit_started = state.exit_start_time_sec if state.exit_start_time_sec is not None else current_time
            if current_time - exit_started >= self.exit_confirm_seconds:
                # Chỉ sau khi ở ngoài đủ lâu mới kết thúc sự kiện và cho phép cảnh báo lần sau.
                self._reset_to_outside(state, current_time)

    def _reset_to_outside(self, state: TrackIntrusionState, current_time: float) -> None:
        # Xóa dữ liệu của phiên xâm nhập vừa kết thúc, nhưng vẫn giữ FSM của track để tiếp tục theo dõi.
        state.entry_time_sec = None
        state.confirmed_time_sec = None
        state.exit_start_time_sec = None
        state.inside_frames = 0
        state.alerted = False
        state.event_id = None
        self._transition(state, IntrusionState.OUTSIDE, current_time)

    @staticmethod
    def _transition(state: TrackIntrusionState, target: IntrusionState, current_time: float) -> None:
        if state.state == target:
            return
        source = state.state
        state.previous_state = source
        state.state = target
        # Lịch sử này giúp kiểm thử và giải thích FSM khi demo mà không tác động quyết định cảnh báo.
        state.state_history.append(StateTransition(current_time, source, target))


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
            intrusion_detector = IntrusionEventDetector(
                confirm_frames=self.confirm_frames,
                source_fps=source_fps,
                frame_stride=self.frame_stride,
            )
            intrusion_count = 0
            frame_index = 0
            processed_frame_count = 0
            # Chỉ gửi một phần frame lên UI để preview nhẹ hơn, tối đa khoảng 10 FPS.
            preview_interval = max(1, int(output_fps / 10))
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
                    video_time_sec = frame_index / source_fps
                    intrusions, inside_count = intrusion_detector.update(
                        detections,
                        zone,
                        video_time_sec=video_time_sec,
                    )

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
                    if self.progress_callback:
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
