"""Tính các thuộc tính hành vi của một người trong vùng cấm.

Đầu vào là vị trí chân, thời gian video và trạng thái FSM của một track.
Mỗi frame cập nhật quãng đường, tốc độ, độ sâu, thời gian đứng yên và đổi hướng.
Sau mỗi 0,5 giây, các giá trị đang tích lũy được đóng thành một FeatureVector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


Point = tuple[float, float]


@dataclass(frozen=True)
class FeatureVector:
    """Một dòng dữ liệu mô tả hành vi của người tại một thời điểm trong video."""

    event_id: str
    track_id: int
    entry_time_sec: float
    confirmed_time_sec: float
    sample_time_sec: float
    elapsed_sec: float
    inside_duration_sec: float
    average_speed_norm: float
    max_speed_norm: float
    max_zone_depth_norm: float
    stopped_duration_sec: float
    reentry_count: int
    direction_change_count: int
    people_in_zone_peak: int
    observation_count: int
    detection_confidence_avg: float


@dataclass
class _TrackFeatureAccumulator:
    """Bộ đếm tạm giữ kết quả của một người trong suốt một lần vào vùng."""

    track_id: int
    entry_time_sec: float
    event_id: str | None = None
    confirmed_time_sec: float | None = None
    last_snapshot_time_sec: float | None = None
    last_position_norm: Point | None = None
    last_position_time_sec: float | None = None
    previous_motion_vector: Point | None = None
    total_distance_norm: float = 0.0
    total_observed_time_sec: float = 0.0
    max_speed_norm: float = 0.0
    max_zone_depth_norm: float = 0.0
    stopped_duration_sec: float = 0.0
    reentry_count: int = 0
    direction_change_count: int = 0
    people_in_zone_peak: int = 0
    observation_count: int = 0
    confidence_sum: float = 0.0


class FeatureExtractor:
    """Theo dõi vị trí chân qua nhiều frame để tính các thuộc tính hành vi."""

    def __init__(
        self,
        snapshot_interval_seconds: float = 0.5,
        stopped_speed_threshold: float = 0.01,
        direction_change_angle_degrees: float = 45.0,
        minimum_motion_norm: float = 0.002,
    ) -> None:
        # Cứ sau khoảng thời gian này, hệ thống tạo một dòng snapshot mới.
        self.snapshot_interval_seconds = max(snapshot_interval_seconds, 0.01)
        # Tốc độ đã chuẩn hóa nhỏ hơn ngưỡng này được xem là đứng yên.
        self.stopped_speed_threshold = max(stopped_speed_threshold, 0.0)
        # Chỉ tính đổi hướng khi góc giữa hướng cũ và hướng mới đủ lớn.
        self.direction_change_cosine = math.cos(math.radians(direction_change_angle_degrees))
        # Bỏ qua dịch chuyển rất nhỏ vì thường chỉ do khung phát hiện bị rung.
        self.minimum_motion_norm = max(minimum_motion_norm, 0.0)
        self._tracks: dict[int, _TrackFeatureAccumulator] = {}

    def observe(
        self,
        *,
        track_id: int,
        event_id: str | None,
        entry_time_sec: float,
        video_time_sec: float,
        foot_point: Point,
        frame_diagonal: float,
        zone_depth_pixels: float,
        people_in_zone: int,
        reentry_count: int,
        detection_confidence: float,
    ) -> FeatureVector | None:
        """Nhận một lần quan sát mới và trả về dòng snapshot khi đủ 0,5 giây."""

        if frame_diagonal <= 0:
            raise ValueError("frame_diagonal must be positive")

        accumulator = self._tracks.get(track_id)
        if accumulator is None or (event_id and accumulator.event_id not in {None, event_id}):
            # Người vừa bước vào vùng: tạo bộ đếm riêng cho lần vào vùng này.
            accumulator = _TrackFeatureAccumulator(track_id=track_id, entry_time_sec=entry_time_sec)
            self._tracks[track_id] = accumulator

        if event_id is not None and accumulator.event_id is None:
            # FSM vừa xác nhận xâm nhập: gắn event_id nhưng vẫn giữ dữ liệu đã đo trước đó.
            accumulator.event_id = event_id
            accumulator.confirmed_time_sec = video_time_sec
            accumulator.last_snapshot_time_sec = video_time_sec

        # Chia tọa độ cho đường chéo ảnh để video 720p và 1080p có thể so sánh với nhau.
        position_norm = (foot_point[0] / frame_diagonal, foot_point[1] / frame_diagonal)
        self._update_motion(accumulator, position_norm, video_time_sec)

        zone_depth_norm = max(zone_depth_pixels, 0.0) / frame_diagonal
        accumulator.max_zone_depth_norm = max(accumulator.max_zone_depth_norm, zone_depth_norm)
        accumulator.people_in_zone_peak = max(accumulator.people_in_zone_peak, people_in_zone)
        accumulator.reentry_count = max(accumulator.reentry_count, reentry_count)
        accumulator.observation_count += 1
        accumulator.confidence_sum += detection_confidence

        if accumulator.event_id is None or accumulator.last_snapshot_time_sec is None:
            # Khi FSM còn chờ xác nhận, chỉ cộng dồn số liệu và chưa ghi snapshot.
            return None
        if video_time_sec - accumulator.last_snapshot_time_sec < self.snapshot_interval_seconds:
            return None

        accumulator.last_snapshot_time_sec = video_time_sec
        return self._build_vector(accumulator, video_time_sec)

    def mark_outside(self, track_id: int) -> None:
        """Ngắt chuỗi chuyển động để không tính bước nhảy qua ngoài polygon thành tốc độ trong vùng."""

        accumulator = self._tracks.get(track_id)
        if accumulator is None:
            return
        accumulator.last_position_norm = None
        accumulator.last_position_time_sec = None
        accumulator.previous_motion_vector = None

    def complete(self, track_id: int, event_id: str, end_time_sec: float) -> FeatureVector | None:
        """Chốt đặc trưng cuối sự kiện và giải phóng bộ nhớ tạm của track."""

        accumulator = self._tracks.get(track_id)
        if accumulator is None or accumulator.event_id != event_id:
            return None
        vector = self._build_vector(accumulator, end_time_sec)
        del self._tracks[track_id]
        return vector

    def discard_unconfirmed(self, track_id: int) -> None:
        """Xóa lần vào vùng bị hủy trước khi FSM xác nhận INTRUDING."""

        accumulator = self._tracks.get(track_id)
        if accumulator is not None and accumulator.event_id is None:
            del self._tracks[track_id]

    def discard_unknown_tracks(self, known_track_ids: set[int]) -> None:
        """Dọn dữ liệu CONFIRMING của track đã hết timeout và bị FSM loại bỏ."""

        for track_id, accumulator in list(self._tracks.items()):
            if track_id not in known_track_ids and accumulator.event_id is None:
                del self._tracks[track_id]

    def _update_motion(
        self,
        accumulator: _TrackFeatureAccumulator,
        position_norm: Point,
        video_time_sec: float,
    ) -> None:
        previous_position = accumulator.last_position_norm
        previous_time = accumulator.last_position_time_sec
        accumulator.last_position_norm = position_norm
        accumulator.last_position_time_sec = video_time_sec
        if previous_position is None or previous_time is None:
            return

        delta_time = video_time_sec - previous_time
        if delta_time <= 0:
            return
        motion_vector = (
            position_norm[0] - previous_position[0],
            position_norm[1] - previous_position[1],
        )
        distance_norm = math.hypot(*motion_vector)
        speed_norm = distance_norm / delta_time
        accumulator.total_distance_norm += distance_norm
        accumulator.total_observed_time_sec += delta_time
        accumulator.max_speed_norm = max(accumulator.max_speed_norm, speed_norm)

        if speed_norm <= self.stopped_speed_threshold:
            accumulator.stopped_duration_sec += delta_time

        if distance_norm >= self.minimum_motion_norm:
            if self._is_direction_change(accumulator.previous_motion_vector, motion_vector):
                accumulator.direction_change_count += 1
            accumulator.previous_motion_vector = motion_vector

    def _is_direction_change(self, previous: Point | None, current: Point) -> bool:
        if previous is None:
            return False
        previous_length = math.hypot(*previous)
        current_length = math.hypot(*current)
        if previous_length < self.minimum_motion_norm or current_length < self.minimum_motion_norm:
            return False
        cosine = (previous[0] * current[0] + previous[1] * current[1]) / (previous_length * current_length)
        return cosine < self.direction_change_cosine

    @staticmethod
    def _build_vector(accumulator: _TrackFeatureAccumulator, sample_time_sec: float) -> FeatureVector:
        if accumulator.event_id is None or accumulator.confirmed_time_sec is None:
            raise RuntimeError("cannot build features before intrusion confirmation")
        average_speed = (
            accumulator.total_distance_norm / accumulator.total_observed_time_sec
            if accumulator.total_observed_time_sec > 0
            else 0.0
        )
        confidence_average = (
            accumulator.confidence_sum / accumulator.observation_count
            if accumulator.observation_count > 0
            else 0.0
        )
        return FeatureVector(
            event_id=accumulator.event_id,
            track_id=accumulator.track_id,
            entry_time_sec=accumulator.entry_time_sec,
            confirmed_time_sec=accumulator.confirmed_time_sec,
            sample_time_sec=sample_time_sec,
            elapsed_sec=max(sample_time_sec - accumulator.entry_time_sec, 0.0),
            inside_duration_sec=accumulator.total_observed_time_sec,
            average_speed_norm=average_speed,
            max_speed_norm=accumulator.max_speed_norm,
            max_zone_depth_norm=accumulator.max_zone_depth_norm,
            stopped_duration_sec=accumulator.stopped_duration_sec,
            reentry_count=accumulator.reentry_count,
            direction_change_count=accumulator.direction_change_count,
            people_in_zone_peak=accumulator.people_in_zone_peak,
            observation_count=accumulator.observation_count,
            detection_confidence_avg=confidence_average,
        )
