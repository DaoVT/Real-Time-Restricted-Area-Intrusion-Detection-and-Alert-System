# Agent Instructions

## Nguyên tắc chung

- Đọc code liên quan trước khi sửa.
- Giữ thay đổi nhỏ, đúng phạm vi yêu cầu.
- Không tự ý refactor lớn khi nhiệm vụ chỉ yêu cầu sửa hẹp.
- Không thay đổi source code nếu nhiệm vụ chỉ yêu cầu tài liệu, báo cáo hoặc phân tích.
- Không thêm dependency mới nếu không thật sự cần thiết.
- Nếu cần dependency mới, phải giải thích lý do, phạm vi sử dụng và cập nhật file cấu hình liên quan.

## API và schema

- Không tự ý đổi REST API schema, WebSocket schema hoặc tên field đã có.
- Nếu bắt buộc sửa schema, phải cập nhật đồng thời:
  - FastAPI endpoint/schema trong `backend/app`.
  - React API/WebSocket client trong `frontend/src`.
  - Tests liên quan trong `tests`.
  - Tài liệu trong `REQUIREMENTS.md`.
- Mọi status WebSocket message phải giữ dạng:

```json
{
  "type": "...",
  "data": {}
}
```

- Frame WebSocket `/ws/jobs/{job_id}/frames` phải gửi binary JPEG bytes, không gửi Base64.

## Backend

- Sử dụng type hints cho code Python mới hoặc code Python được sửa.
- Dùng Pydantic schema cho request/response có cấu trúc.
- Không chạy xử lý video trực tiếp trong HTTP endpoint.
- `POST /api/jobs` phải trả `job_id` nhanh và để xử lý video chạy background.
- Job processing phải hỗ trợ stop qua `threading.Event`.
- Khi xử lý lỗi trong job, phải chuyển job sang `FAILED` và lưu error message.
- Không block FastAPI event loop bằng xử lý CPU/video dài hạn.
- Khi thao tác file/path, ưu tiên `pathlib.Path` để tương thích Windows.
- `VideoCapture` và `VideoWriter` phải được release bằng cleanup rõ ràng.

## Frontend

- Không dùng Redux trong project hiện tại.
- Nếu đổi API hoặc WebSocket payload, phải cập nhật frontend cùng lúc với backend.
- Khi tạo object URL hoặc Blob URL, phải revoke khi không dùng nữa.
- Không gửi tọa độ polygon dạng pixel từ UI; frontend phải gửi normalized coordinates.
- WebSocket reconnect phải có giới hạn, không retry vô hạn.
- Frontend phải xử lý trạng thái backend disconnected.

## Electron

- Luôn giữ `contextIsolation=true`.
- Luôn giữ `nodeIntegration=false`.
- Không expose API Node trực tiếp ra renderer.
- Nếu cần thêm API qua preload, expose qua `contextBridge` với surface nhỏ và rõ ràng.
- Nếu Electron spawn backend process, phải dừng process đó khi app quit.

## YOLO và xử lý video

- YOLO/OpenCV core phải độc lập với FastAPI để có thể chạy qua CLI.
- Chỉ xử lý class `person` trừ khi yêu cầu thay đổi rõ ràng.
- Tracking hiện dùng ByteTrack; không đổi tracker nếu không có lý do kỹ thuật.
- Điểm đại diện người là midpoint cạnh dưới bounding box.
- Cảnh báo chỉ được tạo sau khi nằm trong vùng đủ `confirm_frames` liên tục.
- Không cảnh báo lặp khi cùng `track_id` vẫn đứng trong vùng.
- Phải tiếp tục xử lý job ngay cả khi không có WebSocket client.

## Kiểm thử bắt buộc

Trước khi kết thúc một thay đổi source code, chạy:

```powershell
pytest -q
npm run lint
npm run build
```

Nếu thay đổi chỉ là tài liệu và không chạm source code, có thể không chạy full test, nhưng phải kiểm tra `git diff` để xác nhận chỉ tài liệu thay đổi.

Không che giấu test thất bại. Nếu test/build/lint fail do môi trường hoặc dependency thiếu, báo rõ lệnh, lỗi và phần chưa xác minh.

## Git

- Không commit file runtime như log, PID, output video, evidence image, `backend/data`, `backend/output`, `dist` hoặc `node_modules`.
- Không revert thay đổi của người dùng nếu không được yêu cầu.
- Commit message nên ngắn gọn, mô tả đúng loại thay đổi.
