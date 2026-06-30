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
    stopped = manager.stop_job(job_id)
    assert stopped.status == "STOPPED"

    deadline = time.time() + 2
    while time.time() < deadline:
        snapshot = manager.get_job(job_id)
        if snapshot.status == "STOPPED":
            break
        time.sleep(0.02)
    assert manager.get_job(job_id).status == "STOPPED"
