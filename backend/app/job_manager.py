from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from .schemas import JobCreate, JobSnapshot, JobStatus
from .video_processor import VideoProcessor


@dataclass
class JobRecord:
    snapshot: JobSnapshot
    stop_event: threading.Event
    thread: threading.Thread | None = None


class JobManager:
    def __init__(self, video_dir: Path, output_dir: Path, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.loop = loop
        self._jobs: dict[UUID, JobRecord] = {}
        self._lock = threading.RLock()
        self.status_callback = None
        self.frame_callback = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def create_job(self, payload: JobCreate) -> UUID:
        video_path = self.video_dir / f"{payload.video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError("video_id does not exist")
        job_id = uuid4()
        snapshot = JobSnapshot(
            job_id=job_id,
            video_id=payload.video_id,
            status=JobStatus.PENDING,
            progress=0.0,
        )
        stop_event = threading.Event()
        record = JobRecord(snapshot=snapshot, stop_event=stop_event)
        with self._lock:
            self._jobs[job_id] = record
        thread = threading.Thread(target=self._run_job, args=(job_id, payload, video_path), daemon=True)
        record.thread = thread
        thread.start()
        return job_id

    def get_job(self, job_id: UUID) -> JobSnapshot | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return record.snapshot.model_copy() if record else None

    def stop_job(self, job_id: UUID) -> JobSnapshot | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return None
            record.stop_event.set()
            if record.snapshot.status in {JobStatus.PENDING, JobStatus.RUNNING}:
                record.snapshot.status = JobStatus.STOPPED
            return record.snapshot.model_copy()

    def _update(self, job_id: UUID, **changes) -> JobSnapshot | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return None
            data = record.snapshot.model_dump()
            data.update(changes)
            record.snapshot = JobSnapshot(**data)
            return record.snapshot.model_copy()

    def _emit(self, job_id: UUID, message: dict) -> None:
        if self.status_callback and self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.status_callback(job_id, message), self.loop)

    def _emit_frame(self, job_id: UUID, frame: bytes) -> None:
        if self.frame_callback and self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.frame_callback(job_id, frame), self.loop)

    def _run_job(self, job_id: UUID, payload: JobCreate, video_path: Path) -> None:
        output_dir = self.output_dir / str(job_id)
        self._update(job_id, status=JobStatus.RUNNING)
        self._emit(job_id, {"type": "progress", "data": {"progress": 0.0}})

        def on_progress(stats: dict) -> None:
            events = stats.pop("events", [])
            events = [self._with_evidence_url(job_id, event) for event in events]
            snapshot = self._update(job_id, **stats, status=JobStatus.RUNNING)
            if snapshot:
                self._emit(job_id, {"type": "progress", "data": {"progress": snapshot.progress}})
                self._emit(
                    job_id,
                    {
                        "type": "statistics",
                        "data": {
                            "fps": snapshot.fps,
                            "people_count": snapshot.people_count,
                            "inside_count": snapshot.inside_count,
                            "intrusion_count": snapshot.intrusion_count,
                        },
                    },
                )
            for event in events:
                self._emit(job_id, {"type": "intrusion", "data": event})

        try:
            processor = VideoProcessor(
                input_path=video_path,
                output_dir=output_dir,
                zone_points=[point.model_dump() for point in payload.zone_points],
                confidence=payload.confidence,
                confirm_frames=payload.confirm_frames,
                progress_callback=on_progress,
                frame_callback=lambda frame: self._emit_frame(job_id, frame),
                stop_event=self._jobs[job_id].stop_event,
                process_width=payload.process_width,
                frame_stride=payload.frame_stride,
            )
            result = processor.run()
            status = JobStatus.STOPPED if result.get("stopped") else JobStatus.COMPLETED
            snapshot = self._update(
                job_id,
                status=status,
                progress=1.0 if status == JobStatus.COMPLETED else self.get_job(job_id).progress,
                output_video=result.get("output_video"),
                events_csv=result.get("events_csv"),
                intrusion_count=result.get("intrusion_count", self.get_job(job_id).intrusion_count),
            )
            self._emit(job_id, {"type": "completed", "data": snapshot.model_dump(mode="json") if snapshot else {}})
        except Exception as exc:
            snapshot = self._update(job_id, status=JobStatus.FAILED, error=str(exc))
            self._emit(job_id, {"type": "error", "data": {"message": str(exc), "job": snapshot.model_dump(mode="json") if snapshot else None}})

    def _with_evidence_url(self, job_id: UUID, event: dict) -> dict:
        evidence_path = event.get("evidence_path")
        if not evidence_path:
            return event
        try:
            relative = Path(evidence_path).resolve().relative_to(self.output_dir.resolve())
        except ValueError:
            return event
        enriched = dict(event)
        enriched["evidence_url"] = f"/outputs/{relative.as_posix()}"
        return enriched
