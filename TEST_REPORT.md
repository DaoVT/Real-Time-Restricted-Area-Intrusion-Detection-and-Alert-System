# Test Report

Date: 2026-07-04

## Scope

Requested test type: real end-to-end validation without mocking YOLO, detector, tracker, or `VideoProcessor`.

Required sample inputs:

- `../sample-videos/sample.mp4`
- `../sample-videos/zone.json`

Actual sample input check:

```powershell
Test-Path ..\sample-videos\sample.mp4
Test-Path ..\sample-videos\zone.json
```

Result:

```text
False
False
```

The full video-processing E2E run could not be completed because the required sample video and zone file are not present at the specified paths.

## Commands Run

Documentation/context read:

```powershell
Get-Content REQUIREMENTS.md
Get-Content AGENTS.md
Get-Content README.md
Get-Content TEST_REPORT.md
```

CLI entrypoint:

```powershell
python -m app.cli --help
```

Result: passed after adding the `app.cli` compatibility entrypoint.

YOLO model load:

```powershell
python -c "from backend.app.video_processor import PersonDetector; detector=PersonDetector(0.35); print(type(detector.model).__name__)"
```

Result:

```text
YOLO
```

First run required approval because Ultralytics needed to write settings/model cache outside the sandbox. It created `C:\Users\Admin\AppData\Roaming\Ultralytics\settings.json` and downloaded `yolov8n.pt`. A later sandboxed run succeeded from cache.

Attempted real CLI E2E:

```powershell
python -m app.cli --input ..\sample-videos\sample.mp4 --zone ..\sample-videos\zone.json --output backend\output\e2e
```

Result:

```text
Video processing failed: [Errno 2] No such file or directory: '..\\sample-videos\\zone.json'
```

Automated checks:

```powershell
python -m compileall app backend tests
pytest -q
npm run lint
npm run build
```

## Fixes Made During This QA Pass

The following issues were found before the blocked E2E video run and were fixed:

- `python -m app.cli --help` failed because there was no `app` package. Added `app/__init__.py` and `app/cli.py`.
- Initial job status was `QUEUED`, while the requested E2E lifecycle expects `PENDING -> RUNNING -> COMPLETED`. Changed new jobs and frontend display to `PENDING`.
- No result endpoint existed. Added `GET /api/jobs/{job_id}/result`.
- Evidence images were not displayable in React because only a local file path was sent. Added `/outputs` static serving, `evidence_url` enrichment, and React thumbnail rendering.
- Electron did not manage backend lifecycle. Added backend process spawn on app ready and stop on app quit.
- `yolov8n.pt` was created during YOLO load and is now ignored by Git.

## Files Changed

- `.gitignore`
- `AGENTS.md`
- `README.md`
- `REQUIREMENTS.md`
- `TEST_REPORT.md`
- `app/__init__.py`
- `app/cli.py`
- `backend/app/job_manager.py`
- `backend/app/main.py`
- `backend/app/schemas.py`
- `electron/main.cjs`
- `frontend/src/main.jsx`
- `frontend/src/styles.css`
- `tests/test_api.py`
- `tests/test_job_manager.py`

## Automated Check Results

### Pytest

Command:

```powershell
pytest -q
```

Result:

```text
12 passed in 1.12s
```

### Lint

Command:

```powershell
npm run lint
```

Result: passed with no warnings.

### Build

Command:

```powershell
npm run build
```

Result: passed.

Build artifact summary:

```text
dist/index.html
dist/assets/index-DU97IVtT.css
dist/assets/index-Cuu_-FtY.js
```

## Real E2E Checklist

The items below are not marked passed unless they were actually run against the requested sample video.

1. Model YOLO loads successfully: verified.
2. VideoCapture opens the video: not run; `../sample-videos/sample.mp4` is missing.
3. YOLO only detects class person: not run; sample video is missing.
4. ByteTrack returns track ID: not run; sample video is missing.
5. Normalized coordinates convert correctly to pixels: covered by implementation review and unit coverage indirectly, but not verified against requested `zone.json` because it is missing.
6. Bottom midpoint of bounding box is used as foot position: covered by implementation and unit tests, but not verified on requested video because input is missing.
7. Person must remain inside zone for `confirm_frames` before alerting: covered by unit test, not verified on requested video.
8. One person standing inside does not alert every frame: covered by unit test, not verified on requested video.
9. Person leaving and re-entering creates a new event: not verified on requested video.
10. VideoWriter creates output video that can be opened: not run; sample video/zone missing.
11. Evidence image is created: not run; sample video/zone missing.
12. CSV event file is created and has valid data: not run; sample video/zone missing.
13. Job transitions `PENDING -> RUNNING -> COMPLETED`: code fixed and tests pass with mocked processor, but not verified with real YOLO because sample video/zone missing.
14. WebSocket sends `progress`, `statistics`, `intrusion`, and `completed`: schema/lifecycle tests pass, but not verified with real YOLO because sample video/zone missing.
15. Binary WebSocket sends decodable JPEG: unit test covers JPEG encoding, but not verified with real running job because sample video/zone missing.
16. Result endpoint returns file paths: endpoint added and covered by test with test processor, but not verified with real output files because sample video/zone missing.

## UI/Electron Checklist

These checks could not be completed as real UI E2E because the required sample video and zone file are missing, and no browser automation dependency is configured in the repository.

1. Upload video by React: not run with requested sample.
2. Draw polygon: not run through browser automation.
3. Create job: not run with requested sample.
4. Receive preview: not run with requested sample.
5. Display alert: not run with requested sample.
6. Display evidence image: implementation fixed to render `evidence_url` thumbnails, not verified with real evidence output.
7. Stop a running job: covered by backend tests with test processor, not verified through UI with real YOLO.
8. Run a new job after stop: not verified through UI with real YOLO.
9. Close Electron and verify backend process stops: implementation fixed in `electron/main.cjs`, not verified with a real Electron window in this environment.

## Metrics From Real Requested Sample

These metrics are unavailable because the requested sample files are missing:

- Average FPS: N/A
- People detected: N/A
- Intrusion events: N/A
- Output video path: N/A
- CSV path: N/A
- Evidence image paths: N/A

## Remaining Issues

- `../sample-videos/sample.mp4` is missing.
- `../sample-videos/zone.json` is missing.
- Full no-mock E2E video processing cannot be completed until those files exist at the requested paths.
- Real UI/Electron E2E was not completed; the repository does not currently include browser/Electron automation tooling.
- The source documents contain mojibake in some Vietnamese text when read through PowerShell, likely due encoding display behavior; the files remain readable as Markdown in editors that use UTF-8.
