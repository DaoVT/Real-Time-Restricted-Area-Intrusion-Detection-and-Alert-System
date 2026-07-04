# Requirements

## Mục tiêu project

Project xây dựng một ứng dụng desktop/web để phát hiện người xâm nhập vùng cấm trong video. Người dùng chọn video, vẽ vùng cấm trên frame đầu tiên, upload video lên backend, tạo processing job, theo dõi tiến độ realtime và nhận sự kiện cảnh báo khi một người đứng trong vùng cấm đủ số frame xác nhận.

Kết quả xử lý gồm video có overlay, CSV sự kiện và ảnh bằng chứng cho từng intrusion event.

## Kiến trúc hiện tại

Hệ thống gồm bốn phần chính:

- Electron: tạo cửa sổ desktop, spawn backend FastAPI khi app ready, dừng backend khi app quit, load Vite dev server khi phát triển hoặc file build trong `dist` khi chạy production. Electron đang bật `contextIsolation=true` và `nodeIntegration=false`.
- React + Vite: giao diện chọn video, trích frame đầu, vẽ polygon vùng cấm, upload video, tạo job, nhận trạng thái qua WebSocket JSON và nhận preview frame JPEG binary qua WebSocket riêng.
- FastAPI: cung cấp REST API, WebSocket trạng thái, WebSocket frame preview, quản lý upload video và quản lý processing job.
- YOLO/OpenCV core: xử lý video độc lập trong `backend/app/video_processor.py`, dùng Ultralytics YOLO pretrained model, ByteTrack, OpenCV VideoCapture/VideoWriter, ghi evidence image, CSV và output video.

Các module backend chính:

- `backend/app/main.py`: FastAPI app, REST endpoints, WebSocket endpoints, CORS, lifecycle.
- `backend/app/schemas.py`: Pydantic schemas cho request/response/job/WebSocket message.
- `backend/app/job_manager.py`: JobManager thread-safe, chạy VideoProcessor trong background thread.
- `backend/app/video_processor.py`: PersonDetector, ZoneDetector, IntrusionEventDetector, VideoProcessor và CLI helpers.
- `backend/app/websocket_manager.py`: quản lý WebSocket JSON theo `job_id`.
- `backend/app/frame_manager.py`: quản lý WebSocket binary frame, queue một frame mới nhất.
- `backend/app/cli.py`: CLI độc lập để chạy processor ngoài FastAPI.

## Luồng upload, tạo job, xử lý và cảnh báo

1. Frontend kiểm tra backend qua `GET /health`.
2. Người dùng chọn file video.
3. Frontend tạo object URL, dùng phần tử `video` và `canvas` để lấy frame đầu tiên.
4. Người dùng click trên canvas để tạo polygon vùng cấm bằng tọa độ normalized `x`, `y` trong khoảng `0.0` đến `1.0`.
5. Frontend upload video bằng `POST /api/videos` với multipart field `file`.
6. Backend kiểm tra extension và content type video hợp lệ, kiểm tra file không rỗng, lưu file vào `backend/data/videos/{video_id}.mp4`.
7. Backend trả `video_id`.
8. Frontend gọi `POST /api/jobs` với `video_id`, `confidence`, `confirm_frames`, `zone_points`.
9. Backend tạo `job_id`, lưu snapshot trạng thái `PENDING`, tạo `threading.Event` để stop job, chạy `VideoProcessor` trong background thread và trả `job_id` ngay với HTTP 202.
10. Job chuyển sang `RUNNING`, xử lý từng frame, cập nhật progress/statistics, phát intrusion events và preview frames qua callback.
11. `JobManager` phát message JSON qua `/ws/jobs/{job_id}` và frame JPEG binary qua `/ws/jobs/{job_id}/frames`.
12. Frontend cập nhật progress, FPS, people count, inside count, intrusion count, thêm intrusion event mới lên đầu bảng và hiển thị preview frame realtime.
13. Khi xử lý xong, job chuyển `COMPLETED`. Nếu người dùng stop, job chuyển `STOPPED`. Nếu lỗi, job chuyển `FAILED` và lưu error message.

## REST API hiện có

### `GET /health`

Trả trạng thái backend.

Response:

```json
{
  "status": "ok"
}
```

### `POST /api/videos`

Nhận multipart/form-data với field:

- `file`: video upload.

File hợp lệ hiện tại:

- Extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
- Content types: `video/mp4`, `video/quicktime`, `video/x-msvideo`, `video/x-matroska`, `video/webm`, `application/octet-stream`

Response:

```json
{
  "video_id": "uuid"
}
```

Lỗi:

- `400 unsupported video file`
- `400 empty video file`

### `POST /api/jobs`

Tạo processing job và trả ngay, không xử lý video trực tiếp trong HTTP endpoint.

Request:

```json
{
  "video_id": "uuid",
  "confidence": 0.35,
  "confirm_frames": 5,
  "process_width": 640,
  "frame_stride": 3,
  "save_output_video": false,
  "zone_points": [
    { "x": 0.1, "y": 0.1 },
    { "x": 0.9, "y": 0.1 },
    { "x": 0.5, "y": 0.9 }
  ]
}
```

Validation:

- `confidence`: `0.01` đến `1.0`
- `confirm_frames`: `1` đến `300`
- `process_width`: optional width used for YOLO detection resize, `320` đến `1920`
- `frame_stride`: process every Nth frame, `1` đến `30`
- `save_output_video`: whether to write `result.mp4`; live preview can set this to `false`
- `zone_points`: tối thiểu 3 điểm
- Mỗi điểm normalized có `x`, `y` trong `0.0` đến `1.0`
- Polygon phải có diện tích khác 0

Response status: `202`

```json
{
  "job_id": "uuid"
}
```

Lỗi:

- `404 video_id does not exist`

### `GET /api/jobs/{job_id}`

Trả snapshot hiện tại của job.

Response:

```json
{
  "job_id": "uuid",
  "video_id": "uuid",
  "status": "PENDING",
  "progress": 0.0,
  "fps": 0.0,
  "people_count": 0,
  "inside_count": 0,
  "intrusion_count": 0,
  "error": null,
  "output_video": null,
  "events_csv": null
}
```

`status` có thể là:

- `PENDING`
- `RUNNING`
- `STOPPED`
- `COMPLETED`
- `FAILED`

Lỗi:

- `404 job not found`

### `GET /api/jobs/{job_id}/result`

Trả các đường dẫn kết quả hiện có của job.

Response:

```json
{
  "job_id": "uuid",
  "status": "COMPLETED",
  "output_video": "backend/output/{job_id}/result.mp4",
  "events_csv": "backend/output/{job_id}/events.csv",
  "evidence_images": [
    "backend/output/{job_id}/evidence/{uuid}.jpg"
  ]
}
```

Lỗi:

- `404 job not found`

### `POST /api/jobs/{job_id}/stop`

Đặt stop event cho job. Nếu job đang `PENDING` hoặc `RUNNING`, trạng thái snapshot chuyển sang `STOPPED`.

Response: `JobSnapshot`

Lỗi:

- `404 job not found`

## WebSocket message schema hiện có

### Status WebSocket: `/ws/jobs/{job_id}`

Mọi message JSON phải có dạng:

```json
{
  "type": "...",
  "data": {}
}
```

Các message hiện có:

#### `progress`

```json
{
  "type": "progress",
  "data": {
    "progress": 0.5
  }
}
```

#### `statistics`

```json
{
  "type": "statistics",
  "data": {
    "fps": 25.0,
    "people_count": 1,
    "inside_count": 1,
    "intrusion_count": 1
  }
}
```

#### `intrusion`

```json
{
  "type": "intrusion",
  "data": {
    "frame": 7,
    "track_id": 3,
    "confidence": 0.91,
    "evidence_path": "backend/output/{job_id}/evidence/{uuid}.jpg",
    "evidence_url": "/outputs/{job_id}/evidence/{uuid}.jpg"
  }
}
```

#### `completed`

```json
{
  "type": "completed",
  "data": {
    "job_id": "uuid",
    "video_id": "uuid",
    "status": "COMPLETED",
    "progress": 1.0,
    "fps": 25.0,
    "people_count": 0,
    "inside_count": 0,
    "intrusion_count": 1,
    "error": null,
    "output_video": "backend/output/{job_id}/result.mp4",
    "events_csv": "backend/output/{job_id}/events.csv"
  }
}
```

#### `error`

```json
{
  "type": "error",
  "data": {
    "message": "error message",
    "job": {}
  }
}
```

### Frame WebSocket: `/ws/jobs/{job_id}/frames`

Sends binary JPEG bytes, not Base64.

Frame preview rules:

- Resize preview to maximum 640x360.
- JPEG encode with OpenCV.
- Send at most approximately 10 FPS based on source FPS.
- Per-client queue size is 1.
- If a client is slow, the old frame is dropped and replaced by the newest frame.
- Processing continues even when no frame clients are connected.

## Logic vùng cấm

- Frontend stores polygon points as normalized coordinates in `[0.0, 1.0]`.
- Backend converts normalized points to pixel coordinates using the current video width and height.
- `ZoneDetector.contains()` uses `cv2.pointPolygonTest`.
- A point on the polygon boundary counts as inside.
- The representative point for each detected person is the midpoint of the bottom edge of the bounding box:

```text
((x1 + x2) / 2, y2)
```

## Logic phát hiện người

- `PersonDetector` loads an Ultralytics YOLO pretrained model, default `yolov8n.pt`.
- Detection/tracking uses `model.track(...)`.
- Tracking uses `tracker="bytetrack.yaml"`.
- Only class `0` is processed, corresponding to `person` in COCO models.
- Each detection keeps `track_id`, `bbox` and `confidence`.
- Fast Mode can resize detection frames with `process_width` and skip detection frames with `frame_stride`; detections are scaled back to original frame coordinates before zone checks and drawing.
- YOLO device selection prefers GPU device `0` when PyTorch CUDA is available and falls back to CPU otherwise.

## Logic chống cảnh báo lặp

`IntrusionEventDetector` tracks two structures:

- `_inside_counts`: số frame liên tục mà mỗi `track_id` nằm trong vùng cấm.
- `_alerted_tracks`: các `track_id` đã cảnh báo trong lần đứng trong vùng hiện tại.

Rules:

- Một người chỉ tạo intrusion event khi midpoint cạnh dưới bounding box nằm trong polygon liên tục ít nhất `confirm_frames`.
- Khi một `track_id` đã cảnh báo và vẫn tiếp tục đứng trong vùng, không cảnh báo lặp.
- Khi track rời vùng, inside count reset về 0 và track được phép cảnh báo lại nếu sau đó vào vùng đủ `confirm_frames`.
- Khi track biến mất khỏi frame, state của track đó bị xóa.

## File đầu ra

Upload input:

- `backend/data/videos/{video_id}.mp4`

Per-job output:

- `backend/output/{job_id}/result.mp4`: video có overlay bounding boxes, midpoint và polygon vùng cấm.
- `backend/output/{job_id}/events.csv`: CSV sự kiện với columns `frame`, `track_id`, `confidence`, `evidence_path`.
- `backend/output/{job_id}/evidence/{uuid}.jpg`: ảnh bằng chứng khi có intrusion event.

Output files are served by FastAPI under `/outputs`.

CLI output mặc định:

- `backend/output/cli`

## CLI hiện có

Chạy processor độc lập:

```powershell
python -m backend.app.cli --input sample.mp4 --zone zone.json
```

Options:

- `--input`: đường dẫn video input.
- `--zone`: file JSON chứa danh sách điểm normalized hoặc object có key `points`/`zone_points`.
- `--output`: thư mục output, mặc định `backend/output/cli`.
- `--confidence`: confidence threshold, mặc định `0.35`.
- `--confirm-frames`: số frame xác nhận, mặc định `5`.
- `--process-width`: optional resize width for faster detection, ví dụ `640`.
- `--frame-stride`: process every Nth frame, ví dụ `3`.
- `--no-output-video`: skip writing `result.mp4` for lighter live preview processing.

## Phạm vi không thực hiện hiện tại

Các phần sau chưa có hoặc không thuộc phạm vi hiện tại:

- Không có authentication/authorization.
- Không có database; job state chỉ nằm trong memory.
- Không persist job state qua restart backend.
- Không có queue worker/process pool riêng; mỗi job chạy trong background thread.
- Không có live camera/RTSP input.
- Không có cấu hình model qua UI.
- Không có endpoint phục vụ trực tiếp evidence image hoặc output video cho frontend.
- Không có packaging Electron production hoàn chỉnh.
- Không có multi-user access control.
- Không có retry/resume job sau lỗi.
- Không có giới hạn dung lượng upload theo cấu hình riêng.
- Không có test end-to-end với model YOLO thật và sample video thật trong repository.
