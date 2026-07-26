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
- `backend/app/risk/feature_extractor.py`: tính các đặc trưng hành vi normalized theo từng sự kiện xâm nhập.
- `backend/app/risk/dataset_exporter.py`: ghi event cuối và snapshot định kỳ vào CSV phục vụ train mức cao.
- `backend/app/risk/predictor.py`: load Random Forest `risk_model.joblib` một lần và dự đoán `LOW/MEDIUM/HIGH` cho từng snapshot.
- `backend/app/websocket_manager.py`: quản lý WebSocket JSON theo `job_id`.
- `backend/app/frame_manager.py`: quản lý WebSocket binary frame, queue một frame mới nhất.
- `backend/app/cli.py`: CLI độc lập để chạy processor ngoài FastAPI.
- `high_level_training/data/collect_dataset.py`: chạy nhiều video từ manifest, cô lập detector/ByteTrack theo từng video.
- `high_level_training/data/prepare_virat.py`: quét video VIRAT theo folder camera, ghép zone và tạo manifest.
- `high_level_training/data/prepare_dataset.py`: gộp event và snapshot CSV từ nhiều job.
- `high_level_training/labeling/propagate_event_labels.py`: tạo bản snapshot mới mang nhãn cuối của event cho baseline.
- `high_level_training/evaluation/audit_dataset.py`: kiểm tra schema, giá trị, liên kết event-snapshot và phân bố nhãn.
- `high_level_training/modeling/train_random_forest.py`: kiểm tra nhãn, chia theo camera/video, train và đánh giá Random Forest.

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
11. Khi `FeatureExtractor` tạo snapshot, `RiskPredictor` dùng 8 đặc trưng hành vi để dự đoán `LOW/MEDIUM/HIGH` cùng xác suất.
12. `JobManager` phát message JSON qua `/ws/jobs/{job_id}` và frame JPEG binary qua `/ws/jobs/{job_id}/frames`.
13. Frontend cập nhật progress, thống kê, mức rủi ro, xác suất, intrusion event và preview frame realtime.
14. Khi xử lý xong, job chuyển `COMPLETED`. Nếu người dùng stop, job chuyển `STOPPED`. Nếu lỗi, job chuyển `FAILED` và lưu error message.

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

Ràng buộc validate:

- `confidence`: `0.01` đến `1.0`
- `confirm_frames`: `1` đến `300`
- `process_width`: chiều rộng tùy chọn dùng để resize khi YOLO detect, `320` đến `1920`
- `frame_stride`: xử lý mỗi N frame, `1` đến `30`
- `save_output_video`: có ghi `result.mp4` hay không; live preview có thể đặt giá trị này là `false`
- `zone_points`: tối thiểu 3 điểm
- Mỗi điểm normalized có `x`, `y` trong `0.0` đến `1.0`
- Polygon phải có diện tích khác 0

HTTP response status: `202`

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
  "risk_model_available": true,
  "risk_model_error": null,
  "risk_level": null,
  "risk_track_id": null,
  "risk_probabilities": {},
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
    "intrusion_count": 1,
    "risk_model_available": true,
    "risk_model_error": null
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

#### `risk`

Message được gửi mỗi khi có snapshot hành vi mới:

```json
{
  "type": "risk",
  "data": {
    "available": true,
    "event_id": "uuid",
    "track_id": 3,
    "video_time_sec": 12.5,
    "level": "HIGH",
    "probabilities": {
      "LOW": 0.01,
      "MEDIUM": 0.09,
      "HIGH": 0.9
    },
    "model_trained_at_utc": "2026-07-24T00:00:00+00:00"
  }
}
```

Nếu model thiếu hoặc dự đoán lỗi, message `risk` có `available=false` và `error`.
Lỗi riêng của model risk không được làm dừng YOLO, FSM hoặc processing job.

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

Gửi binary JPEG bytes, không gửi Base64.

Quy tắc preview frame:

- Resize preview về tối đa 640x360.
- Encode JPEG bằng OpenCV.
- Gửi tối đa xấp xỉ 10 FPS dựa trên FPS của video nguồn.
- Mỗi client có queue size bằng 1.
- Nếu client chậm, frame cũ bị bỏ và được thay bằng frame mới nhất.
- Job xử lý vẫn tiếp tục ngay cả khi không có client nhận frame.

## Logic vùng cấm

- Frontend lưu các điểm polygon bằng tọa độ normalized trong `[0.0, 1.0]`.
- Backend chuyển tọa độ normalized sang tọa độ pixel theo chiều rộng và chiều cao hiện tại của video.
- `ZoneDetector.contains()` dùng `cv2.pointPolygonTest`.
- Điểm nằm trên biên polygon vẫn được tính là ở trong vùng.
- Điểm đại diện cho mỗi người được phát hiện là midpoint của cạnh dưới bounding box:

```text
((x1 + x2) / 2, y2)
```

## Logic phát hiện người

- `PersonDetector` tải Ultralytics YOLO pretrained model, mặc định là `yolov8n.pt`.
- Phát hiện/tracking dùng `model.track(...)`.
- Tracking dùng `tracker="bytetrack.yaml"`.
- Chỉ xử lý class `0`, tương ứng với `person` trong COCO models.
- Mỗi detection giữ `track_id`, `bbox` và `confidence`.
- Fast Mode có thể resize frame detect bằng `process_width` và bỏ qua frame bằng `frame_stride`; detection được scale ngược về tọa độ frame gốc trước khi kiểm tra vùng và vẽ overlay.
- Chọn device YOLO ưu tiên GPU device `0` khi PyTorch CUDA khả dụng, nếu không thì fallback về CPU.

## FSM và logic chống cảnh báo lặp

`IntrusionEventDetector` giữ một `TrackIntrusionState` riêng cho mỗi `track_id`. FSM gồm:

- `OUTSIDE`: track ở ngoài vùng cấm.
- `CONFIRMING`: điểm chân đã vào polygon nhưng chưa đủ thời gian xác nhận.
- `INTRUDING`: đã xác nhận xâm nhập và đã tạo cảnh báo.
- `EXITING`: điểm chân vừa ra ngoài, đang chờ xác nhận để chống rung ở biên polygon.
- `LOST`: ByteTrack tạm thời không trả về track; trạng thái trước đó được giữ để có thể khôi phục.

Luồng chính:

```text
OUTSIDE -> CONFIRMING -> INTRUDING -> EXITING -> OUTSIDE
                         ^              |
                         +--------------+
```

Rules:

- Một người chỉ tạo intrusion event khi midpoint cạnh dưới bounding box nằm trong polygon đủ `confirm_frames`.
- `confirm_frames` được quy đổi theo `source_fps` và `frame_stride`, sau đó so sánh bằng thời gian video gốc thay vì thời gian chạy của máy.
- Khi một track vẫn ở `INTRUDING`, hệ thống không tạo cảnh báo lặp.
- Khi điểm chân ra ngoài, track chuyển sang `EXITING`; chỉ khi ngoài vùng liên tục đủ `0.3` giây video mới chuyển về `OUTSIDE`.
- Nếu track quay lại vùng trong lúc `EXITING`, track trở về `INTRUDING` và không cảnh báo lại.
- Track mất dấu chuyển sang `LOST`. Nếu cùng `track_id` xuất hiện lại trước `2.0` giây video, FSM khôi phục trạng thái trước; nếu hết timeout, state được dọn khỏi memory.
- Mỗi lần chuyển trạng thái được lưu trong `state_history` của track để kiểm thử và debug.
- FSM xác nhận có xâm nhập hay không. Pipeline hiện đã trích và xuất đặc trưng hành vi; model đánh giá rủi ro `LOW/MEDIUM/HIGH` là thành phần độc lập và chưa được train/tích hợp.

## Dữ liệu xử lý mức cao

- `FeatureExtractor` bắt đầu tích lũy vị trí chân từ trạng thái `CONFIRMING`, nhưng chỉ xuất dữ liệu sau khi FSM tạo `event_id` ở trạng thái `INTRUDING`.
- Tốc độ được chuẩn hóa theo đường chéo frame và thời gian video gốc để không phụ thuộc độ phân giải hoặc tốc độ chạy của máy.
- Độ sâu vùng là khoảng cách dương từ điểm chân tới biên polygon, chuẩn hóa theo đường chéo frame.
- Chuỗi vị trí bị ngắt khi người ra ngoài hoặc track chuyển `LOST`, tránh tính khoảng che khuất thành tốc độ giả.
- Snapshot được ghi sau mỗi `0.5` giây video trong khi sự kiện diễn ra.
- Event cuối được chốt khi FSM xác nhận `OUTSIDE`, track hết timeout, video kết thúc hoặc job bị stop.
- Các cột `label` và `annotator_note` để trống để gán nhãn thủ công sau khi xem evidence/video.
- Pipeline mức cao không train lại và không thay đổi model YOLO `best.pt` của xử lý mức trung.
- Random Forest chỉ dùng đặc trưng hành vi; `detection_confidence_avg` và `observation_count` không nằm trong feature train ban đầu.
- Dataset phải có ít nhất ba camera/video độc lập để tạo train, validation và test; mọi snapshot cùng `event_id` phải nằm trong cùng split.
- Dataset VIRAT được lưu ngoài repository, tổ chức video theo `camera_id` và dùng một zone normalized cho mỗi góc quay.
- Có thể gán nhãn cuối cho toàn event rồi truyền xuống snapshot để tạo baseline, hoặc gán từng snapshot cho bài toán rủi ro realtime; không trộn hai cách trong cùng một dataset.

## File đầu ra

Upload input:

- `backend/data/videos/{video_id}.mp4`

Per-job output:

- `backend/output/{job_id}/result.mp4`: video có overlay bounding boxes, midpoint và polygon vùng cấm.
- `backend/output/{job_id}/events.csv`: CSV sự kiện với columns `frame`, `track_id`, `confidence`, `evidence_path`.
- `backend/output/{job_id}/evidence/{uuid}.jpg`: ảnh bằng chứng khi có intrusion event.
- `backend/output/{job_id}/high_level_events.csv`: một dòng đặc trưng tổng hợp cho mỗi sự kiện đã kết thúc.
- `backend/output/{job_id}/high_level_snapshots.csv`: đặc trưng tại các thời điểm đang diễn ra để train dự đoán realtime.

`high_level_events.csv` giữ `camera_id` do người thu thập tự đặt và `evidence_path` để đối chiếu khi gán nhãn.

Các file output được FastAPI phục vụ dưới route `/outputs`.

CLI output mặc định:

- `backend/output/cli`

## CLI hiện có

Chạy processor độc lập:

```powershell
python -m backend.app.cli --input sample.mp4 --zone zone.json
```

Các option:

- `--input`: đường dẫn video input.
- `--zone`: file JSON chứa danh sách điểm normalized hoặc object có key `points`/`zone_points`.
- `--output`: thư mục output, mặc định `backend/output/cli`.
- `--confidence`: confidence threshold, mặc định `0.35`.
- `--confirm-frames`: số frame xác nhận, mặc định `5`.
- `--process-width`: chiều rộng resize tùy chọn để detect nhanh hơn, ví dụ `640`.
- `--frame-stride`: xử lý mỗi N frame, ví dụ `3`.
- `--no-output-video`: bỏ qua bước ghi `result.mp4` để live preview nhẹ hơn.
- `--model`: đường dẫn weights dùng cho lần chạy CLI, mặc định `best.pt`.
- `--camera-id`: mã nguồn/góc quay tự quy ước cho dataset mức cao.

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
- Chưa có model mức cao phân loại rủi ro `LOW/MEDIUM/HIGH`; hiện tại hệ thống đã có FSM và pipeline xuất đặc trưng nhưng chưa có dataset đã gán nhãn để train.
- Chưa có GlobalTrackManager để ghép một người khi ByteTrack cấp `track_id` mới và chưa có Re-ID ngoại hình.
