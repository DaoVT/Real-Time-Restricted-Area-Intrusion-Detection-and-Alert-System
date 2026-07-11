from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class Point(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class JobCreate(BaseModel):
    video_id: UUID
    confidence: float = Field(default=0.35, ge=0.01, le=1.0)
    confirm_frames: int = Field(default=5, ge=1, le=300)
    process_width: int | None = Field(default=None, ge=320, le=1920)
    frame_stride: int = Field(default=1, ge=1, le=30)
    save_output_video: bool = True
    filter_type: str = Field(default="auto")
    zone_points: list[Point] = Field(min_length=3)

    @field_validator("zone_points")
    @classmethod
    def polygon_must_have_area(cls, points: list[Point]) -> list[Point]:
      area = 0.0
      for index, point in enumerate(points):
          other = points[(index + 1) % len(points)]
          area += point.x * other.y - other.x * point.y
      if abs(area) < 0.0001:
          raise ValueError("zone polygon must have area")
      return points


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobSnapshot(BaseModel):
    job_id: UUID
    video_id: UUID
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    fps: float = 0.0
    people_count: int = 0
    inside_count: int = 0
    intrusion_count: int = 0
    error: str | None = None
    output_video: str | None = None
    events_csv: str | None = None


class VideoUploadResponse(BaseModel):
    video_id: UUID


class JobCreateResponse(BaseModel):
    job_id: UUID


class JobResultResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    output_video: str | None = None
    events_csv: str | None = None
    evidence_images: list[str] = Field(default_factory=list)


class WsMessage(BaseModel):
    type: Literal["progress", "statistics", "intrusion", "completed", "error"]
    data: dict[str, Any]
