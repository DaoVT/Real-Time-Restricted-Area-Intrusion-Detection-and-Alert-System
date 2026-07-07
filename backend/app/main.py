from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .frame_manager import FrameConnectionManager
from .job_manager import JobManager
from .schemas import JobCreate, JobCreateResponse, JobResultResponse, JobSnapshot, VideoUploadResponse
from .websocket_manager import ConnectionManager

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "output"
# Thư mục runtime này được tạo tự động và không nên commit vào git.
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
    "application/octet-stream",
}

app = FastAPI(title="Restricted Area Intrusion Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

connections = ConnectionManager()
frame_connections = FrameConnectionManager()
jobs = JobManager(VIDEO_DIR, OUTPUT_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Gắn JobManager với event loop hiện tại để worker thread gửi được WebSocket message.
    jobs.set_loop(asyncio.get_running_loop())
    jobs.status_callback = connections.broadcast
    jobs.frame_callback = frame_connections.publish
    yield


app.router.lifespan_context = lifespan


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/videos", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)) -> VideoUploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS or (file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES):
        raise HTTPException(status_code=400, detail="unsupported video file")
    video_id = uuid4()
    jobs.video_dir.mkdir(parents=True, exist_ok=True)
    # Lưu bằng UUID để tránh đụng tên file người dùng và tránh lộ path gốc.
    path = jobs.video_dir / f"{video_id}.mp4"
    first_chunk = await file.read(1024 * 1024)
    if not first_chunk:
        raise HTTPException(status_code=400, detail="empty video file")
    with path.open("wb") as output:
        output.write(first_chunk)
        while chunk := await file.read(1024 * 1024):
            output.write(chunk)
    return VideoUploadResponse(video_id=video_id)


@app.post("/api/jobs", response_model=JobCreateResponse, status_code=202)
async def create_job(payload: JobCreate) -> JobCreateResponse:
    # Endpoint chỉ tạo job; xử lý YOLO luôn chạy nền trong JobManager.
    try:
        job_id = jobs.create_job(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JobCreateResponse(job_id=job_id)


@app.get("/api/jobs/{job_id}", response_model=JobSnapshot)
async def get_job(job_id: UUID) -> JobSnapshot:
    snapshot = jobs.get_job(job_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="job not found")
    return snapshot


@app.get("/api/jobs/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: UUID) -> JobResultResponse:
    snapshot = jobs.get_job(job_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="job not found")
    job_output_dir = jobs.output_dir / str(job_id)
    # Result trả path file để phục vụ kiểm thử và mở lại output sau khi job hoàn tất.
    evidence_images = sorted(str(path) for path in (job_output_dir / "evidence").glob("*.jpg"))
    return JobResultResponse(
        job_id=job_id,
        status=snapshot.status,
        output_video=snapshot.output_video,
        events_csv=snapshot.events_csv,
        evidence_images=evidence_images,
    )


@app.post("/api/jobs/{job_id}/stop", response_model=JobSnapshot)
async def stop_job(job_id: UUID) -> JobSnapshot:
    snapshot = jobs.stop_job(job_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="job not found")
    return snapshot


@app.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: UUID) -> None:
    # WebSocket JSON chỉ gửi trạng thái/thống kê/sự kiện, theo schema {type, data}.
    await connections.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await connections.disconnect(job_id, websocket)


@app.websocket("/ws/jobs/{job_id}/frames")
async def frame_websocket(websocket: WebSocket, job_id: UUID) -> None:
    # WebSocket frame gửi binary JPEG bytes để preview nhẹ hơn Base64.
    try:
        await frame_connections.frame_sender(job_id, websocket)
    except WebSocketDisconnect:
        return
