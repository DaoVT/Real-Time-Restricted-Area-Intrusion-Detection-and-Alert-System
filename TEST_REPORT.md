# Test Report

Date: 2026-06-30

## Scope

- `REQUIREMENTS.md` and `AGENTS.md` were requested but are not present in this workspace.
- Implemented and checked the project against the task prompt supplied in the chat.

## Services

- Backend started with `python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`.
- Backend health check passed: `GET /health` returned `{"status":"ok"}`.
- Frontend started with `npm run dev -- --host 127.0.0.1`.
- Frontend responded on `http://127.0.0.1:5173` with HTTP 200.

## Automated Checks

- `python -m compileall backend tests`: passed.
- `pytest -q`: 10 passed.
- `npm run lint`: passed with no warnings.
- `npm run build`: passed.

## API And WebSocket Review

- React upload flow calls `POST /api/videos` with multipart form data field `file`.
- React job creation payload matches FastAPI `JobCreate`: `video_id`, `confidence`, `confirm_frames`, `zone_points`.
- React connects to `/ws/jobs/{job_id}` and expects `progress`, `statistics`, `intrusion`, `completed`, and `error`.
- All status WebSocket messages use `{ "type": "...", "data": {} }`.
- React connects to `/ws/jobs/{job_id}/frames` with `binaryType = "blob"` and does not use Base64.

## QA Checklist

- JobManager uses an `RLock` around job state and a `threading.Event` for stop.
- `POST /api/jobs` starts processing in a background thread and returns immediately.
- Stop job path is covered by tests with a mock slow processor.
- No-person/no-event behavior is covered by `IntrusionEventDetector` tests.
- WebSocket message schema and connect/disconnect lifecycle are covered by tests.
- Frame preview queue keeps only the newest frame and drops stale frames for slow clients.
- Windows paths are handled with `pathlib.Path`.
- `VideoCapture` and `VideoWriter` are released in `finally`.
- React revokes Blob URLs for first-frame previews and live JPEG frames.
- Electron uses `contextIsolation=true` and `nodeIntegration=false`.

## Manual YOLO/CLI Check

Command attempted:

```powershell
python -m backend.app.cli --input sample.mp4 --zone zone.json
```

Result: failed because `zone.json` and `sample.mp4` are not present in the workspace. Dependencies including `ultralytics`, `torch`, and `opencv-python` are installed, but a real sample video and zone file are still needed for an end-to-end YOLO run.

## Runtime Notes

- Backend PID is written to `backend.pid`.
- Frontend PID is written to `frontend.pid`.
- Runtime logs are ignored by Git.
