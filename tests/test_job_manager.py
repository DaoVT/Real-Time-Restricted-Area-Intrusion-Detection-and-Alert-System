from __future__ import annotations

import time
from uuid import uuid4

from backend.app.job_manager import JobManager
from backend.app.schemas import JobCreate


class SlowProcessor:
    def __init__(self, *args, progress_callback=None, stop_event=None, **kwargs):
        self.progress_callback = progress_callback
        self.stop_event = stop_event

    def run(self):
        for index in range(20):
            if self.stop_event.is_set():
                return {"stopped": True, "intrusion_count": 0}
            if self.progress_callback:
                self.progress_callback({"progress": index / 20, "fps": 10, "people_count": 0, "inside_count": 0, "intrusion_count": 0, "events": []})
            time.sleep(0.01)
        return {"stopped": False, "output_video": "out.mp4", "events_csv": "events.csv", "intrusion_count": 0}


class RiskProcessor:
    def __init__(self, *args, risk_callback=None, **kwargs):
        self.risk_callback = risk_callback

    def run(self):
        self.risk_callback(
            {
                "available": True,
                "event_id": "event-1",
                "track_id": 4,
                "video_time_sec": 2.5,
                "level": "HIGH",
                "probabilities": {"LOW": 0.02, "MEDIUM": 0.18, "HIGH": 0.8},
            }
        )
        return {"stopped": False, "intrusion_count": 1}


def test_job_manager_stop_is_thread_safe(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.app.job_manager.VideoProcessor", SlowProcessor)
    video_id = uuid4()
    video_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    video_dir.mkdir()
    output_dir.mkdir()
    (video_dir / f"{video_id}.mp4").write_bytes(b"video")

    manager = JobManager(video_dir, output_dir)
    payload = JobCreate(
        video_id=video_id,
        confidence=0.3,
        confirm_frames=2,
        zone_points=[{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1}, {"x": 0.5, "y": 0.9}],
    )
    job_id = manager.create_job(payload)
    assert manager.get_job(job_id).status in {"PENDING", "RUNNING"}
    stopped = manager.stop_job(job_id)
    assert stopped.status == "STOPPED"

    deadline = time.time() + 2
    while time.time() < deadline:
        snapshot = manager.get_job(job_id)
        if snapshot.status == "STOPPED":
            break
        time.sleep(0.02)
    assert manager.get_job(job_id).status == "STOPPED"


def test_job_manager_adds_evidence_url_for_output_images(tmp_path):
    manager = JobManager(tmp_path / "videos", tmp_path / "output")
    job_id = uuid4()
    evidence_path = tmp_path / "output" / str(job_id) / "evidence" / "frame.jpg"
    event = manager._with_evidence_url(job_id, {"evidence_path": str(evidence_path)})
    assert event["evidence_url"] == f"/outputs/{job_id}/evidence/frame.jpg"


def test_job_manager_stores_latest_risk_prediction(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.app.job_manager.VideoProcessor", RiskProcessor)
    video_id = uuid4()
    video_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    video_dir.mkdir()
    output_dir.mkdir()
    (video_dir / f"{video_id}.mp4").write_bytes(b"video")
    manager = JobManager(video_dir, output_dir, risk_predictor=object())
    job_id = manager.create_job(
        JobCreate(
            video_id=video_id,
            confidence=0.3,
            confirm_frames=2,
            zone_points=[
                {"x": 0.1, "y": 0.1},
                {"x": 0.9, "y": 0.1},
                {"x": 0.5, "y": 0.9},
            ],
        )
    )

    deadline = time.time() + 2
    while time.time() < deadline:
        snapshot = manager.get_job(job_id)
        if snapshot and snapshot.status == "COMPLETED":
            break
        time.sleep(0.01)

    snapshot = manager.get_job(job_id)
    assert snapshot.risk_model_available is True
    assert snapshot.risk_level == "HIGH"
    assert snapshot.risk_track_id == 4
    assert snapshot.risk_probabilities["HIGH"] == 0.8
