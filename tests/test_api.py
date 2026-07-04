from __future__ import annotations

import time

from fastapi.testclient import TestClient

from backend.app import main as app_module
from backend.app.job_manager import JobManager


class DummyProcessor:
    def __init__(self, *args, progress_callback=None, frame_callback=None, stop_event=None, **kwargs):
        self.progress_callback = progress_callback
        self.frame_callback = frame_callback
        self.stop_event = stop_event

    def run(self):
        if self.progress_callback:
            self.progress_callback(
                {
                    "progress": 0.5,
                    "fps": 25.0,
                    "people_count": 1,
                    "inside_count": 1,
                    "intrusion_count": 1,
                    "events": [{"frame": 7, "track_id": 3, "confidence": 0.91, "evidence_path": "evidence/a.jpg"}],
                }
            )
        if self.frame_callback:
            self.frame_callback(b"\xff\xd8\xff\xd9")
        return {"stopped": bool(self.stop_event and self.stop_event.is_set()), "output_video": "result.mp4", "events_csv": "events.csv", "intrusion_count": 1}


def install_test_manager(monkeypatch, tmp_path):
    manager = JobManager(tmp_path / "videos", tmp_path / "output")
    manager.video_dir.mkdir(parents=True)
    manager.output_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "jobs", manager)
    return manager


def test_health():
    with TestClient(app_module.app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_result_endpoint_returns_paths(monkeypatch, tmp_path):
    install_test_manager(monkeypatch, tmp_path)
    monkeypatch.setattr("backend.app.job_manager.VideoProcessor", DummyProcessor)

    with TestClient(app_module.app) as client:
        upload = client.post("/api/videos", files={"file": ("sample.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")})
        video_id = upload.json()["video_id"]
        created = client.post(
            "/api/jobs",
            json={
                "video_id": video_id,
                "confidence": 0.4,
                "confirm_frames": 2,
                "zone_points": [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1}, {"x": 0.5, "y": 0.9}],
            },
        )
        job_id = created.json()["job_id"]
        deadline = time.time() + 2
        while time.time() < deadline:
            snapshot = client.get(f"/api/jobs/{job_id}").json()
            if snapshot["status"] == "COMPLETED":
                break
            time.sleep(0.02)

        result = client.get(f"/api/jobs/{job_id}/result")
        assert result.status_code == 200
        assert result.json()["output_video"] == "result.mp4"
        assert result.json()["events_csv"] == "events.csv"


def test_upload_video_validation(monkeypatch, tmp_path):
    install_test_manager(monkeypatch, tmp_path)
    with TestClient(app_module.app) as client:
        bad = client.post("/api/videos", files={"file": ("note.txt", b"hello", "text/plain")})
        assert bad.status_code == 400

        good = client.post("/api/videos", files={"file": ("sample.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")})
        assert good.status_code == 200
        assert "video_id" in good.json()


def test_create_get_and_stop_job(monkeypatch, tmp_path):
    install_test_manager(monkeypatch, tmp_path)
    monkeypatch.setattr("backend.app.job_manager.VideoProcessor", DummyProcessor)

    with TestClient(app_module.app) as client:
        upload = client.post("/api/videos", files={"file": ("sample.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")})
        video_id = upload.json()["video_id"]
        payload = {
            "video_id": video_id,
            "confidence": 0.4,
            "confirm_frames": 2,
            "zone_points": [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1}, {"x": 0.5, "y": 0.9}],
        }
        created = client.post("/api/jobs", json=payload)
        assert created.status_code == 202
        job_id = created.json()["job_id"]

        deadline = time.time() + 2
        snapshot = None
        while time.time() < deadline:
            snapshot = client.get(f"/api/jobs/{job_id}").json()
            if snapshot["status"] in {"COMPLETED", "FAILED"}:
                break
            time.sleep(0.02)
        assert snapshot["status"] == "COMPLETED"
        assert snapshot["intrusion_count"] == 1

        stopped = client.post(f"/api/jobs/{job_id}/stop")
        assert stopped.status_code == 200


def test_create_job_rejects_missing_video(monkeypatch, tmp_path):
    install_test_manager(monkeypatch, tmp_path)
    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/jobs",
            json={
                "video_id": "00000000-0000-0000-0000-000000000000",
                "confidence": 0.4,
                "confirm_frames": 2,
                "zone_points": [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1}, {"x": 0.5, "y": 0.9}],
            },
        )
        assert response.status_code == 404
