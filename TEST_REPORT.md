# Test Report

Date: 2026-07-04

## Scope

Requested test type: real end-to-end validation without mocking YOLO, detector, tracker, or `VideoProcessor`.

Sample inputs used:

- `..\sample-videos\sample.mp4`
- `..\sample-videos\zone.json`

Resolved paths:

- `D:\sample-videos\sample.mp4`
- `D:\sample-videos\zone.json`

## Commands Run

CLI help:

```powershell
python -m app.cli --help
```

YOLO model load:

```powershell
python -c "from backend.app.video_processor import PersonDetector; detector=PersonDetector(0.35); print(type(detector.model).__name__)"
```

Result:

```text
YOLO
```

Real CLI E2E:

```powershell
python -m app.cli --input ..\sample-videos\sample.mp4 --zone ..\sample-videos\zone.json --output backend\output\e2e-rerun
```

Result:

```text
{'stopped': False, 'output_video': 'backend\\output\\e2e-rerun\\result.mp4', 'events_csv': 'backend\\output\\e2e-rerun\\events.csv', 'intrusion_count': 9}
```

Fast Mode CLI E2E:

```powershell
python -m app.cli --input ..\sample-videos\sample.mp4 --zone ..\sample-videos\zone.json --output backend\output\e2e-fast2 --process-width 640 --frame-stride 3
```

Result:

```text
{'stopped': False, 'output_video': 'backend\\output\\e2e-fast2\\result.mp4', 'events_csv': 'backend\\output\\e2e-fast2\\events.csv', 'intrusion_count': 6}
```

Live no-output-video E2E with CUDA venv:

```powershell
D:\venvs\intrusion\Scripts\python.exe -m app.cli --input ..\sample-videos\sample.mp4 --zone ..\sample-videos\zone.json --output backend\output\e2e-live-no-video --process-width 640 --frame-stride 3 --no-output-video
```

Result:

```text
{'stopped': False, 'output_video': None, 'events_csv': 'backend\\output\\e2e-live-no-video\\events.csv', 'intrusion_count': 6}
```

Output inspection:

```powershell
python -c "... OpenCV/CSV output inspection ..."
```

Automated checks:

```powershell
pytest -q
npm run lint
npm run build
```

## Fixes Made

- Fixed `zone.json` UTF-8 BOM loading by reading zone files with `encoding="utf-8-sig"`.
- Added a regression test for BOM encoded zone files.
- Added `lap>=0.5.12` to `requirements.txt` because Ultralytics ByteTrack requires it.
- Added `sample-videos/` to `.gitignore` so local test videos are not committed.
- Added Fast Mode with `process_width` and `frame_stride` for faster CPU demos.
- Added YOLO device selection: prefer GPU `0` when CUDA is available, otherwise CPU.
- Added `save_output_video` / `--no-output-video` so live preview jobs can skip MP4 writing.

## Files Changed

- `.gitignore`
- `TEST_REPORT.md`
- `backend/app/video_processor.py`
- `requirements.txt`
- `tests/test_video_processor.py`
- `backend/app/cli.py`
- `backend/app/job_manager.py`
- `backend/app/schemas.py`
- `frontend/src/main.jsx`
- `frontend/src/styles.css`
- `README.md`
- `REQUIREMENTS.md`

## Automated Check Results

### Pytest

Command:

```powershell
pytest -q
```

Result:

```text
13 passed in 2.62s
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

## Real E2E Output

Output folder:

```text
backend\output\e2e-rerun
```

Output video:

```text
backend\output\e2e-rerun\result.mp4
```

Video inspection:

```text
video_exists True
video_size 98286886
video_opened True
frames 4548
fps 59.94
```

CSV:

```text
backend\output\e2e-rerun\events.csv
rows 9
```

First CSV event:

```text
frame=227
track_id=1
confidence=0.8893588781356812
evidence_path=backend\output\e2e-rerun\evidence\e19477d3-8638-4a74-8feb-0644a2a21df6.jpg
```

Evidence:

```text
evidence_count 9
first_evidence_decodable True
first_evidence_shape (1080, 1920, 3)
```

## Fast Mode E2E Output

Output folder:

```text
backend\output\e2e-fast2
```

Fast Mode settings:

```text
process_width 640
frame_stride 3
```

Output inspection:

```text
video_opened True
frames 1516
fps 19.98
csv_rows 6
evidence_count 6
```

Fast Mode processed fewer frames and produced fewer intrusion events than accurate mode. This is expected: it trades some temporal precision for faster CPU processing.

## Live No-Output-Video E2E Output

Output folder:

```text
backend\output\e2e-live-no-video
```

Settings:

```text
process_width 640
frame_stride 3
no_output_video true
python D:\venvs\intrusion\Scripts\python.exe
torch 2.6.0+cu124
cuda NVIDIA GeForce MX550
```

Output inspection:

```text
result_exists False
csv_rows 6
evidence_count 6
first_event_frame 313
first_event_track_id 1
```

This mode skipped MP4 writing and completed substantially faster than the previous Fast Mode run that wrote `result.mp4`.

## Real E2E Checklist

- YOLO model loads successfully: verified.
- VideoCapture opens the sample video: verified by successful processing.
- YOLO only detects class person: verified by implementation using `classes=[0]`; runtime completed with YOLO.
- ByteTrack returns track ID: verified by CSV rows containing `track_id`.
- Normalized zone coordinates are loaded and converted to pixel polygon: verified by successful processing with `zone.json`.
- Bottom midpoint of bounding box is used as foot position: verified by implementation and tests.
- Person must remain inside zone for `confirm_frames` before alerting: verified by implementation and tests.
- One person standing inside does not alert every frame: verified by 9 events across 4548 frames.
- Person leaving and re-entering creates a new event: multiple events were produced; manual visual confirmation of leave/re-enter is still pending.
- VideoWriter creates an output video that can be opened: verified.
- Evidence images are created: verified.
- CSV event file is created and has valid data: verified.

## Still Pending

- Real API WebSocket E2E with YOLO-backed JobManager.
- Manual React UI run: upload video, draw polygon, create job, receive preview, show alerts, show evidence image.
- Manual Electron run: verify closing Electron stops its managed backend process.
- Visual tuning of `zone.json`; the polygon was estimated from a screenshot and may need adjustment in the UI for the exact video frame.
