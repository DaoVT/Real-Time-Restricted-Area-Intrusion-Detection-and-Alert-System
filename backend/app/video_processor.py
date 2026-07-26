"""Lõi xử lý video dùng chung cho CLI và backend.

Luồng xử lý:
1. OpenCV đọc từng khung hình.
2. YOLO phát hiện người, ByteTrack gán ID theo dõi.
3. FSM kiểm tra điểm chân và xác nhận người xâm nhập vùng cấm.
4. FeatureExtractor tính các thuộc tính hành vi của từng sự kiện.
5. RiskPredictor dùng các thuộc tính đó để dự đoán LOW/MEDIUM/HIGH.
6. Kết quả được lưu vào CSV, ảnh bằng chứng, video và gửi lên giao diện.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any, Callable, Iterable
from uuid import uuid4

import cv2
import numpy as np

from .image_filters import process_frame
from .risk import DatasetExporter, FeatureExtractor, RiskPredictor
from .schemas import Point


ProgressCallback = Callable[[dict], None]
FrameCallback = Callable[[bytes], None]
RiskCallback = Callable[[dict[str, Any]], None]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def resolve_yolo_model_path(model_name: str | Path | None = None) -> Path:
    """Chọn file trọng số YOLO có sẵn mà không âm thầm tải model khi chạy job."""
    configured_model = model_name or os.getenv("YOLO_MODEL_PATH")
    if configured_model:
        requested_path = Path(configured_model).expanduser()
        candidates = [requested_path]
        if not requested_path.is_absolute():
            candidates.extend([Path.cwd() / requested_path, PROJECT_ROOT / requested_path])
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"Không tìm thấy model YOLO: {configured_model}")

    # Ưu tiên model đã train mức trung; nếu chưa có thì dùng model YOLO pretrained của project.
    for filename in ("best.pt", "yolov8n.pt"):
        candidate = PROJECT_ROOT / filename
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Không tìm thấy model YOLO. Hãy đặt best.pt hoặc yolov8n.pt ở thư mục gốc, "
        "hoặc cấu hình biến môi trường YOLO_MODEL_PATH."
    )


class PersonDetector:
    def __init__(self, confidence: float = 0.15, model_name: str = "best.pt") -> None:
        self.confidence = confidence
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics is not installed or cannot be imported") from exc
        self.model_path = resolve_yolo_model_path(model_name)
        self.model = YOLO(str(self.model_path))
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

    def depth_pixels(self, point: tuple[float, float]) -> float:
        # Khoảng cách dương tới biên polygon biểu diễn người đã tiến sâu vào vùng bao nhiêu pixel.
        return max(float(cv2.pointPolygonTest(self.points, point, True)), 0.0)

    def draw(self, frame: np.ndarray, is_active: bool = False) -> None:
        color = (0, 0, 255) if is_active else (0, 180, 255)
        # Vẽ viền ngoài
        cv2.polylines(frame, [self.points], isClosed=True, color=color, thickness=2)
        # Vẽ phần tô màu bán trong suốt bên trong
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.points], color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)


# ==========================================================
# MÁY TRẠNG THÁI XÂM NHẬP (FSM)
# Mỗi người có một trạng thái riêng. FSM chỉ quyết định khi nào một lần
# đi vào vùng được xem là sự kiện; FSM chưa dự đoán LOW/MEDIUM/HIGH.
# ==========================================================
class IntrusionState(str, Enum):
    # OUTSIDE: điểm chân đang nằm ngoài polygon vùng cấm.
    OUTSIDE = "OUTSIDE"
    # CONFIRMING: điểm chân đã vào vùng nhưng còn chờ đủ thời gian xác nhận.
    CONFIRMING = "CONFIRMING"
    # INTRUDING: đã đủ thời gian, sự kiện được tạo và cảnh báo đúng một lần.
    INTRUDING = "INTRUDING"
    # EXITING: điểm chân vừa ra ngoài, chờ thêm để chắc chắn người đã rời vùng.
    EXITING = "EXITING"
    # LOST: ByteTrack tạm mất người; FSM giữ trạng thái cũ trong thời gian chờ.
    LOST = "LOST"


@dataclass
class StateTransition:
    # Một bản ghi cho biết FSM đổi từ trạng thái nào sang trạng thái nào tại thời gian video nào.
    video_time_sec: float
    from_state: IntrusionState
    to_state: IntrusionState


@dataclass(frozen=True)
class CompletedIntrusion:
    # Thông tin tối thiểu để FeatureExtractor chốt một dòng sự kiện khi FSM kết thúc phiên xâm nhập.
    track_id: int
    event_id: str
    entry_time_sec: float
    confirmed_time_sec: float
    end_time_sec: float


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
    reentry_count: int = 0
    event_id: str | None = None
    last_foot_point: tuple[float, float] | None = None
    state_history: list[StateTransition] = field(default_factory=list)


class IntrusionEventDetector:
    """Quản lý FSM riêng cho từng track_id và trả về cảnh báo mới."""

    def __init__(
        self,
        confirm_frames: int,
        source_fps: float = 25.0,
        frame_stride: int = 1,
        exit_confirm_seconds: float = 0.3,
        track_timeout_seconds: float = 2.0,
    ) -> None:
        # Giá trị tối thiểu bảo đảm máy trạng thái luôn có ngưỡng hợp lệ.
        self.confirm_frames = max(confirm_frames, 1)
        self.source_fps = max(source_fps, 1.0)
        self.frame_stride = max(frame_stride, 1)
        self.exit_confirm_seconds = max(exit_confirm_seconds, 0.0)
        self.track_timeout_seconds = max(track_timeout_seconds, 0.0)
        # Ví dụ track_states[7] là toàn bộ trạng thái của người có track_id 7.
        self.track_states: dict[int, TrackIntrusionState] = {}
        self._completed_intrusions: list[CompletedIntrusion] = []
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
        # BƯỚC 1: Track biến mất khỏi frame hiện tại được chuyển tạm sang LOST.
        current_time = self._resolve_video_time(video_time_sec)
        active_tracks = {detection.track_id for detection in detections}
        self._mark_missing_tracks(active_tracks, current_time)

        # BƯỚC 2: Kiểm tra điểm chân và cập nhật trạng thái riêng của từng người.
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

            # Không dùng tâm cơ thể. Trung điểm cạnh dưới bounding box đại diện cho chân.
            if zone.contains(detection.bottom_midpoint):
                inside_count += 1
                if self._update_inside(state, current_time):
                    new_intrusions.append(detection)
            else:
                self._update_outside(detection.track_id, state, current_time)

        return new_intrusions, inside_count

    def drain_completed_intrusions(self) -> list[CompletedIntrusion]:
        """Lấy các sự kiện vừa kết thúc rồi xóa hàng đợi để không xuất CSV trùng lặp."""

        completed = self._completed_intrusions
        self._completed_intrusions = []
        return completed

    def finish(self, video_time_sec: float) -> list[CompletedIntrusion]:
        """Chốt các sự kiện còn mở khi video hết hoặc người dùng dừng job."""

        for track_id, state in self.track_states.items():
            self._complete_intrusion(track_id, state, video_time_sec)
        return self.drain_completed_intrusions()

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
                # Sự kiện kết thúc tại lúc bắt đầu mất dấu, không cộng thêm thời gian chờ timeout.
                self._complete_intrusion(track_id, state, lost_since)
                del self.track_states[track_id]

    def _update_inside(self, state: TrackIntrusionState, current_time: float) -> bool:
        """Xử lý trường hợp điểm chân của người đang nằm trong vùng cấm."""

        # Có mặt trong vùng nên tăng bộ đếm trong và xóa bộ đếm ngoài.
        state.inside_frames += 1
        state.outside_frames = 0

        if state.state == IntrusionState.OUTSIDE:
            # Mới bước vào: ghi thời điểm bắt đầu và chờ xác nhận, chưa cảnh báo.
            state.entry_time_sec = current_time
            state.entry_count += 1
            state.reentry_count = 0
            state.alerted = False
            state.event_id = None
            self._transition(state, IntrusionState.CONFIRMING, current_time)

        if state.state == IntrusionState.EXITING:
            # Quay lại khi vẫn còn trong thời gian chờ ra: tiếp tục sự kiện cũ.
            state.exit_start_time_sec = None
            state.reentry_count += 1
            self._transition(state, IntrusionState.INTRUDING, current_time)
            return False

        entry_time = state.entry_time_sec if state.entry_time_sec is not None else current_time
        if state.state == IntrusionState.CONFIRMING and current_time - entry_time >= self.confirm_seconds:
            # Chỉ đoạn này trả True nên một lần vào vùng chỉ tạo một cảnh báo.
            state.confirmed_time_sec = current_time
            state.alerted = True
            state.event_id = str(uuid4())
            self._transition(state, IntrusionState.INTRUDING, current_time)
            return True
        return False

    def _update_outside(self, track_id: int, state: TrackIntrusionState, current_time: float) -> None:
        """Xử lý trường hợp điểm chân của người đang nằm ngoài vùng cấm."""

        # Có mặt ngoài vùng nên tăng bộ đếm ngoài và xóa bộ đếm trong.
        state.outside_frames += 1
        state.inside_frames = 0

        if state.state == IntrusionState.CONFIRMING:
            # Chưa đủ thời gian xác nhận mà đã ra: hủy lần vào vùng này.
            self._reset_to_outside(track_id, state, current_time)
            return
        if state.state == IntrusionState.INTRUDING:
            # Đã xâm nhập nhưng vừa ra: chờ 0,3 giây để tránh điểm chân rung ở biên.
            state.exit_start_time_sec = current_time
            self._transition(state, IntrusionState.EXITING, current_time)
            return
        if state.state == IntrusionState.EXITING:
            exit_started = state.exit_start_time_sec if state.exit_start_time_sec is not None else current_time
            if current_time - exit_started >= self.exit_confirm_seconds:
                # Ở ngoài đủ lâu: kết thúc sự kiện và cho phép cảnh báo ở lần vào sau.
                self._reset_to_outside(track_id, state, exit_started)

    def _reset_to_outside(self, track_id: int, state: TrackIntrusionState, current_time: float) -> None:
        # Xóa dữ liệu của phiên xâm nhập vừa kết thúc, nhưng vẫn giữ FSM của track để tiếp tục theo dõi.
        self._complete_intrusion(track_id, state, current_time)
        state.entry_time_sec = None
        state.confirmed_time_sec = None
        state.exit_start_time_sec = None
        state.inside_frames = 0
        state.alerted = False
        state.event_id = None
        self._transition(state, IntrusionState.OUTSIDE, current_time)

    def _complete_intrusion(
        self,
        track_id: int,
        state: TrackIntrusionState,
        end_time_sec: float,
    ) -> None:
        if state.event_id is None or state.entry_time_sec is None or state.confirmed_time_sec is None:
            return
        self._completed_intrusions.append(
            CompletedIntrusion(
                track_id=track_id,
                event_id=state.event_id,
                entry_time_sec=state.entry_time_sec,
                confirmed_time_sec=state.confirmed_time_sec,
                end_time_sec=end_time_sec,
            )
        )
        # Xóa event_id ngay sau khi đưa vào hàng đợi để finish() hoặc timeout không chốt lại lần hai.
        state.event_id = None

    @staticmethod
    def _transition(state: TrackIntrusionState, target: IntrusionState, current_time: float) -> None:
        if state.state == target:
            return
        source = state.state
        state.previous_state = source
        state.state = target
        # Lịch sử này giúp kiểm thử và giải thích FSM khi demo mà không tác động quyết định cảnh báo.
        state.state_history.append(StateTransition(current_time, source, target))


# ==========================================================
# LUỒNG GHÉP MỨC TRUNG VÀ MỨC CAO
# VideoProcessor gọi lần lượt YOLO -> ByteTrack -> FSM -> trích thuộc tính
# -> Random Forest, sau đó lưu file và gửi callback cho JobManager.
# ==========================================================
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
        camera_id: str = "",
        risk_predictor: RiskPredictor | None = None,
        risk_callback: RiskCallback | None = None,
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
        # camera_id là mã nguồn/góc quay do người thu thập dataset tự đặt, không phải serial camera thật.
        self.camera_id = camera_id
        self.risk_predictor = risk_predictor
        self.risk_callback = risk_callback

    def run(self) -> dict:
        # BƯỚC 1: Tạo thư mục riêng cho CSV, ảnh bằng chứng và video của job.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir = self.output_dir / "evidence"
        evidence_dir.mkdir(exist_ok=True)
        events_csv = self.output_dir / "events.csv"
        output_video = self.output_dir / "result.mp4"

        # BƯỚC 2: OpenCV mở video và chuẩn bị đọc lần lượt từng frame.
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

            # BƯỚC 3: Tạo bộ phát hiện người và đổi polygon từ 0..1 sang tọa độ ảnh.
            detector = self.detector or PersonDetector(self.confidence)
            zone = ZoneDetector(self.zone_points, width, height)
            # BƯỚC 4: Tạo FSM để theo dõi một lần vào/ra vùng của từng track_id.
            intrusion_detector = IntrusionEventDetector(
                confirm_frames=self.confirm_frames,
                source_fps=source_fps,
                frame_stride=self.frame_stride,
            )
            # BƯỚC 5: Tạo bộ tính thuộc tính. Phần này chỉ đọc kết quả, không train lại YOLO.
            feature_extractor = FeatureExtractor(snapshot_interval_seconds=0.5)
            frame_diagonal = math.hypot(width, height)
            video_id = self.input_path.stem
            event_evidence_paths: dict[str, str] = {}
            risk_error_reported = False
            intrusion_count = 0
            frame_index = 0
            processed_frame_count = 0
            # Chỉ gửi một phần frame lên UI để preview nhẹ hơn, tối đa khoảng 10 FPS.
            preview_interval = max(1, int(output_fps / 10))
            # Gửi dữ liệu thống kê lên UI khoảng 5 lần/giây để tránh làm giật giao diện web
            stat_interval = max(1, int(output_fps / 5))
            started_at = time.perf_counter()

            with (
                events_csv.open("w", newline="", encoding="utf-8") as csv_file,
                DatasetExporter(self.output_dir) as dataset_exporter,
            ):
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

                    # MỨC TRUNG: YOLO tìm người, ByteTrack cấp track_id qua nhiều frame.
                    detections = scale_detections(detector.track(detect_frame), scale_x, scale_y)
                    # MỨC CAO - PHẦN LUẬT: FSM xác nhận người có thực sự xâm nhập không.
                    video_time_sec = frame_index / source_fps
                    intrusions, inside_count = intrusion_detector.update(
                        detections,
                        zone,
                        video_time_sec=video_time_sec,
                    )

                    # MỨC CAO - PHẦN DỮ LIỆU: cộng dồn tốc độ, thời gian, độ sâu và hành vi.
                    for detection in detections:
                        track_state = intrusion_detector.track_states[detection.track_id]
                        is_inside = zone.contains(detection.bottom_midpoint)
                        if is_inside and track_state.entry_time_sec is not None:
                            snapshot = feature_extractor.observe(
                                track_id=detection.track_id,
                                event_id=track_state.event_id,
                                entry_time_sec=track_state.entry_time_sec,
                                video_time_sec=video_time_sec,
                                foot_point=detection.bottom_midpoint,
                                frame_diagonal=frame_diagonal,
                                zone_depth_pixels=zone.depth_pixels(detection.bottom_midpoint),
                                people_in_zone=inside_count,
                                reentry_count=track_state.reentry_count,
                                detection_confidence=detection.confidence,
                            )
                            if snapshot is not None:
                                # Gắn mã camera vào từng snapshot để có thể chia tập train/validation/test
                                # theo nguồn video, tránh cùng một góc quay xuất hiện ở nhiều tập.
                                dataset_exporter.write_snapshot(
                                    video_id,
                                    snapshot,
                                    camera_id=self.camera_id,
                                )
                                # MỨC CAO - PHẦN MODEL: dùng 8 thuộc tính để dự đoán LOW/MEDIUM/HIGH.
                                if self.risk_predictor is not None and self.risk_callback is not None:
                                    try:
                                        self.risk_callback(self.risk_predictor.predict(snapshot))
                                    except Exception as exc:
                                        # Phân loại mức cao là phần bổ sung; lỗi model không được
                                        # làm dừng YOLO, FSM hoặc việc ghi kết quả video.
                                        if not risk_error_reported:
                                            self.risk_callback(
                                                {
                                                    "available": False,
                                                    "error": str(exc),
                                                }
                                            )
                                            risk_error_reported = True
                                        self.risk_predictor = None
                        else:
                            feature_extractor.mark_outside(detection.track_id)

                    # Khi FSM kết thúc một event, chốt một dòng tổng hợp vào high_level_events.csv.
                    for completed in intrusion_detector.drain_completed_intrusions():
                        final_features = feature_extractor.complete(
                            completed.track_id,
                            completed.event_id,
                            completed.end_time_sec,
                        )
                        if final_features is not None:
                            dataset_exporter.write_event(
                                video_id,
                                final_features,
                                camera_id=self.camera_id,
                                evidence_path=event_evidence_paths.pop(completed.event_id, ""),
                            )

                    # Hủy dữ liệu của lần vào vùng chưa đủ điều kiện và dọn track đã hết timeout.
                    for track_id, track_state in intrusion_detector.track_states.items():
                        if track_state.state == IntrusionState.OUTSIDE:
                            feature_extractor.discard_unconfirmed(track_id)
                        elif track_state.state == IntrusionState.LOST:
                            # Ngắt chuỗi vị trí để thời gian bị che khuất không làm sai tốc độ hoặc đứng yên.
                            feature_extractor.mark_outside(track_id)
                    feature_extractor.discard_unknown_tracks(set(intrusion_detector.track_states))

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
                        track_state = intrusion_detector.track_states[detection.track_id]
                        event_id = track_state.event_id or str(uuid4())
                        evidence_path = evidence_dir / f"{event_id}.jpg"
                        cv2.imwrite(str(evidence_path), frame)
                        event_evidence_paths[event_id] = str(evidence_path)
                        event = {
                            "frame": frame_index,
                            "track_id": detection.track_id,
                            "confidence": detection.confidence,
                            "evidence_path": str(evidence_path),
                        }
                        csv_writer.writerow(event)
                        # Ghi ngay xuống đĩa để API/UI kiểm tra được sự kiện khi job vẫn đang chạy.
                        csv_file.flush()
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
                    # Sự kiện phải được gửi ngay. Nếu chỉ gửi theo nhịp thống kê, các cảnh báo
                    # xuất hiện giữa hai nhịp sẽ có ảnh/CSV nhưng bị mất khỏi giao diện.
                    if self.progress_callback and (
                        intrusion_events or processed_frame_count % stat_interval == 0
                    ):
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

                # Video có thể kết thúc khi người vẫn trong vùng; vẫn phải chốt sự kiện để không mất dữ liệu train.
                final_video_time_sec = frame_index / source_fps
                for completed in intrusion_detector.finish(final_video_time_sec):
                    final_features = feature_extractor.complete(
                        completed.track_id,
                        completed.event_id,
                        completed.end_time_sec,
                    )
                    if final_features is not None:
                        dataset_exporter.write_event(
                            video_id,
                            final_features,
                            camera_id=self.camera_id,
                            evidence_path=event_evidence_paths.pop(completed.event_id, ""),
                        )

            return {
                # Kết quả cuối cùng được JobManager lưu vào snapshot của job.
                "stopped": self.stop_event.is_set(),
                "output_video": str(output_video) if self.save_output_video else None,
                "events_csv": str(events_csv),
                "high_level_events_csv": str(self.output_dir / "high_level_events.csv"),
                "high_level_snapshots_csv": str(self.output_dir / "high_level_snapshots.csv"),
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
