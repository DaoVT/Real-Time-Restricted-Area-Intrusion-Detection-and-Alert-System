# Real-Time Restricted Area Intrusion Detection and Alert System

Ứng dụng desktop/web dùng để upload video, vẽ vùng cấm, phát hiện người xâm nhập vùng cấm bằng YOLO + ByteTrack, xem preview realtime và lưu bằng chứng cảnh báo.



Luồng chính của hệ thống:

1. Người dùng chọn video trên giao diện React/Electron.
2. Người dùng vẽ polygon vùng cấm trên frame đầu tiên.
3. Frontend upload video lên FastAPI backend.
4. Backend tạo job xử lý nền, không xử lý video trực tiếp trong HTTP endpoint.
5. YOLO phát hiện người, ByteTrack gán `track_id`.
6. Hệ thống lấy điểm giữa cạnh dưới bounding box làm vị trí chân người.
7. FSM riêng của từng `track_id` quản lý các trạng thái `OUTSIDE`, `CONFIRMING`, `INTRUDING`, `EXITING` và `LOST`.
8. Nếu chân người nằm trong vùng cấm đủ `confirm_frames` thì FSM tạo một cảnh báo; rung ở biên vùng hoặc mất dấu ngắn không tạo cảnh báo lặp.
9. `FeatureExtractor` thu thập tốc độ, độ sâu vùng, thời gian đứng yên và các đặc trưng hành vi để tạo dataset mức cao.
10. Frontend nhận progress, thống kê, event cảnh báo và preview frame qua WebSocket.

## Cấu Trúc

- `backend/app`: FastAPI API, WebSocket, JobManager và lõi xử lý video.
- `backend/app/risk`: trích đặc trưng và xuất dataset hành vi `LOW/MEDIUM/HIGH`.
- `training`: pipeline train YOLO mức trung hiện có; phần mức cao không sửa folder này.
- `high_level_training`: công cụ gộp dataset và hướng dẫn gán nhãn model mức cao.
- `frontend/src`: giao diện React + Vite.
- `electron`: Electron main process và preload.
- `tests`: pytest cho API, JobManager, WebSocket và VideoProcessor.
- `backend/data`: video upload runtime, không commit.
- `backend/output`: output video, CSV và ảnh bằng chứng runtime, không commit.

Mỗi job tạo thêm `high_level_events.csv` và `high_level_snapshots.csv`. Xem hướng dẫn tại `high_level_training/README.md`.

Khi thu thập dữ liệu bằng CLI, dùng `--model` để chọn weights và `--camera-id` để đặt mã nguồn/góc quay. Hai tùy chọn này không thay đổi pipeline train YOLO trong `training/`.

## Yêu Cầu Cài Đặt

Máy cần có:

- Python 3.11+ hoặc Python 3.13 đang dùng trong project.
- Node.js 18+.
- npm.
- GPU NVIDIA nếu muốn chạy nhanh hơn bằng CUDA.
- Dung lượng trống đủ lớn, vì PyTorch CUDA có thể tải vài GB.

Nếu ổ C ít dung lượng, nên tạo virtual environment ở ổ D.

## Cài Backend Python

### 1. Tạo virtual environment

Khuyến nghị tạo venv ở ổ D:

```powershell
python -m venv D:\venvs\intrusion
D:\venvs\intrusion\Scripts\Activate.ps1
```

Kiểm tra đã vào đúng venv:

```powershell
python -c "import sys; print(sys.executable)"
```

Kết quả nên là:

```text
D:\venvs\intrusion\Scripts\python.exe
```

Nếu vẫn hiện Python ở ổ C như bên dưới thì đang chạy sai môi trường:

```text
C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe
```

### 2. Cài PyTorch

Nếu chỉ chạy CPU:

```powershell
pip install torch torchvision torchaudio
```

Nếu dùng GPU NVIDIA với CUDA, cài bản CUDA của PyTorch. Ví dụ bản CUDA 12.4:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Kiểm tra PyTorch và CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Nếu có GPU và cài đúng, kết quả sẽ có:

```text
True
NVIDIA ...
```

### 3. Cài dependency backend

```powershell
pip install -r requirements.txt
```

Kiểm tra YOLO import được:

```powershell
python -c "import ultralytics; import torch; print('ultralytics ok'); print(torch.cuda.is_available())"
```

## Cài Frontend/Electron

```powershell
npm install
```

## Cấu hình model mức cao

Backend mặc định tìm model tại:

```text
backend\models\risk_model.joblib
```

Copy model v5 đã train vào vị trí mặc định:

```powershell
New-Item -ItemType Directory -Force backend\models
Copy-Item D:\datasets\high-level-model-v5\risk_model.joblib backend\models\risk_model.joblib
```

Hoặc dùng model ở vị trí khác:

```powershell
$env:RISK_MODEL_PATH="D:\datasets\high-level-model-v5\risk_model.joblib"
```

File `.joblib` là artifact nhị phân và không được commit. Nếu model không tồn tại hoặc
không load được, YOLO/FSM vẫn xử lý video nhưng giao diện báo model rủi ro chưa sẵn sàng.

## Chạy Project Khi Phát Triển

Mở terminal 1 để chạy backend bằng đúng Python trong venv:

```powershell
D:\venvs\intrusion\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Kiểm tra backend:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Kết quả:

```json
{
  "status": "ok"
}
```

Mở terminal 2 để chạy frontend:

```powershell
npm run dev
```

Mở trình duyệt:

```text
http://127.0.0.1:5173
```

Chạy Electron:

```powershell
npm run electron
```

Lưu ý: nếu Electron tự spawn backend bằng lệnh `python`, nó có thể dùng nhầm Python ổ C. Khi cần chắc chắn dùng đúng venv, hãy chạy backend thủ công bằng lệnh ở trên trước.

## Chạy CLI Xử Lý Video

Ví dụ:

```powershell
D:\venvs\intrusion\Scripts\python.exe -m backend.app.cli --input ..\sample-videos\sample.mp4 --zone ..\sample-videos\zone.json --output backend\output\e2e
```

Chạy Fast Mode:

```powershell
D:\venvs\intrusion\Scripts\python.exe -m backend.app.cli --input ..\sample-videos\sample.mp4 --zone ..\sample-videos\zone.json --output backend\output\e2e --process-width 640 --frame-stride 3
```

Chạy Fast Mode và không ghi video output:

```powershell
D:\venvs\intrusion\Scripts\python.exe -m backend.app.cli --input ..\sample-videos\sample.mp4 --zone ..\sample-videos\zone.json --output backend\output\e2e --process-width 640 --frame-stride 3 --no-output-video
```

## File Sample Video

Project kỳ vọng sample nằm ngoài thư mục project:

```text
..\sample-videos\sample.mp4
..\sample-videos\zone.json
```

Tức là nếu project ở:

```text
D:\Real-Time Restricted Area Intrusion Detection and Alert System
```

thì sample nên đặt ở:

```text
D:\sample-videos\sample.mp4
D:\sample-videos\zone.json
```

## Chạy Kiểm Tra

```powershell
pytest -q
npm run lint
npm run build
```

## Ghi Chú

- Backend REST API chạy ở `http://127.0.0.1:8000`.
- Frontend Vite chạy ở `http://127.0.0.1:5173`.
- WebSocket status: `/ws/jobs/{job_id}`.
- Message `risk` gửi mức `LOW/MEDIUM/HIGH` và xác suất theo từng snapshot hành vi.
- WebSocket preview frame: `/ws/jobs/{job_id}/frames`.
- Preview frame gửi binary JPEG, không dùng Base64.
- Không commit `backend/data`, `backend/output`, `dist`, `node_modules`, file log hoặc file PID.
