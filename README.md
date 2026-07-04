# Real-Time Restricted Area Intrusion Detection and Alert System

Desktop/web application for uploading a video, drawing a restricted zone, processing person intrusions with YOLO, and viewing live job statistics.

## Structure

- `backend/app`: FastAPI API, WebSocket managers, job manager, and video processing core.
- `frontend/src`: React + Vite UI.
- `electron`: Electron main process and preload script.
- `tests`: pytest coverage for REST APIs, job behavior, and WebSocket lifecycle.

## Development

```powershell
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
npm install
npm run dev
```

Run checks:

```powershell
pytest
npm run lint
npm run build
```

Run the standalone processor:

```powershell
python -m app.cli --input sample.mp4 --zone zone.json
```

The backend implementation is also available directly:

```powershell
python -m backend.app.cli --input sample.mp4 --zone zone.json
```
