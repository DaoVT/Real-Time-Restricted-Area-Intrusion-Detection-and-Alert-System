# Quy trình VIRAT cho xử lý mức cao

Tài liệu này hướng dẫn tạo dataset `LOW/MEDIUM/HIGH` từ video VIRAT Ground. Quy trình không
train lại YOLO và không thay đổi phần xử lý mức trung trong `training/`.

## 1. Chuẩn bị dữ liệu ngoài repository

Tải VIRAT Ground từ [nguồn chính thức](https://viratdata.org/) và tuân thủ thỏa thuận sử dụng.
Không commit video hoặc annotation VIRAT lên GitHub.

Sắp xếp mỗi góc quay vào một folder riêng:

```text
D:\datasets\virat\
├── raw\
│   ├── VIRAT_S_000001\
│   │   ├── clip_001.mp4
│   │   └── clip_002.mp4
│   ├── VIRAT_S_000002\
│   │   └── clip_003.mp4
│   └── VIRAT_S_000003\
│       └── clip_004.mp4
├── zones\
│   ├── VIRAT_S_000001.json
│   ├── VIRAT_S_000002.json
│   └── VIRAT_S_000003.json
├── manifests\
├── outputs\
└── dataset\
```

Tên folder cấp đầu là `camera_id`. Các video có cùng góc quay phải nằm trong cùng folder và
dùng chung một file zone.

## 2. Tạo zone cho từng camera

Mỗi file zone chứa ít nhất ba điểm normalized trong `[0, 1]`:

```json
{
  "camera_id": "VIRAT_S_000001",
  "points": [
    {"x": 0.20, "y": 0.35},
    {"x": 0.70, "y": 0.35},
    {"x": 0.85, "y": 0.85},
    {"x": 0.10, "y": 0.85}
  ]
}
```

`load_zone_file()` chỉ đọc `points`; `camera_id` trong JSON dùng để người gán nhãn dễ kiểm tra.

## 3. Tạo manifest tự động

```powershell
D:\venvs\intrusion\Scripts\python.exe -m high_level_training.data.prepare_virat `
  --raw-root D:\datasets\virat\raw `
  --zones-root D:\datasets\virat\zones `
  --output D:\datasets\virat\manifests\pilot.csv `
  --videos-per-camera 3
```

Công cụ sẽ:

- Tìm video `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.
- Lấy `camera_id` từ folder cấp đầu.
- Ghép `zones\<camera_id>.json`.
- Dừng và báo rõ nếu camera chưa có zone.
- Tạo `output_name` riêng cho từng video.
- Có thể giới hạn đều số video trên mỗi camera bằng `--videos-per-camera`.

Nếu video nằm trực tiếp trong `raw`, tên chuẩn như
`VIRAT_S_010204_00_000060_000120.mp4` sẽ được suy ra camera `VIRAT_S_0102`.
Hai chữ số `04` là số đoạn ghi cùng góc camera, không được dùng làm `camera_id`; nhóm đúng theo
camera tránh để cùng một bối cảnh lọt vào cả train và test.

## 4. Kiểm tra pilot trước khi chạy YOLO

```powershell
D:\venvs\intrusion\Scripts\python.exe -m high_level_training.data.collect_dataset `
  --manifest D:\datasets\virat\manifests\pilot.csv `
  --output-root D:\datasets\virat\outputs `
  --dry-run `
  --limit 10
```

`--dry-run` chỉ kiểm tra đường dẫn, mã camera và tên output; không tải model AI.

## 5. Thu thập đặc trưng

Sau khi dry-run hợp lệ:

```powershell
D:\venvs\intrusion\Scripts\python.exe -m high_level_training.data.collect_dataset `
  --manifest D:\datasets\virat\manifests\pilot.csv `
  --output-root D:\datasets\virat\outputs `
  --model best.pt `
  --process-width 640 `
  --frame-stride 3 `
  --limit 10
```

Nếu chưa có `best.pt` mức trung, có thể pilot bằng `--model yolov8n.pt`. Tuy nhiên, khi bắt đầu
thu thập dataset chính thức phải cố định một model và một bộ tham số cho mọi video.

Mặc định công cụ không ghi `result.mp4` để tiết kiệm dung lượng. Thêm `--save-output-video`
nếu cần xem lại đầy đủ overlay.

Mỗi output hoàn tất có `.collection_complete.json`. Nếu lệnh bị dừng giữa chừng, marker không
được tạo và collector sẽ không coi CSV dở là dữ liệu hoàn chỉnh.

Mỗi video tạo:

```text
outputs\<output_name>\
├── evidence\
├── events.csv
├── high_level_events.csv
└── high_level_snapshots.csv
```

## 6. Gộp dữ liệu

```powershell
D:\venvs\intrusion\Scripts\python.exe -m high_level_training.data.prepare_dataset `
  --source D:\datasets\virat\outputs `
  --destination D:\datasets\virat\dataset
```

## 7. Gán nhãn

Có hai chế độ, cần chọn một chế độ nhất quán.

### Nhãn cuối cho toàn sự kiện

Phù hợp để tạo baseline nhanh. Xem video/evidence rồi điền `label` và `annotator_note` trong
`high_level_events.csv`. Sau đó truyền nhãn xuống mọi snapshot:

```powershell
D:\venvs\intrusion\Scripts\python.exe -m high_level_training.labeling.propagate_event_labels `
  --events D:\datasets\virat\dataset\high_level_events.csv `
  --snapshots D:\datasets\virat\dataset\high_level_snapshots.csv `
  --output D:\datasets\virat\dataset\high_level_snapshots_labeled.csv
```

File gốc không bị ghi đè. Cách này coi mọi thời điểm trong sự kiện có cùng nhãn cuối.

### Nhãn thay đổi theo thời gian

Phù hợp hơn cho cảnh báo realtime. Gán trực tiếp từng dòng trong `high_level_snapshots.csv`
dựa trên `video_time_sec`. Không chạy công cụ truyền nhãn vì snapshot đầu và cuối có thể mang
mức rủi ro khác nhau.

## 8. Kiểm tra chất lượng

```powershell
D:\venvs\intrusion\Scripts\python.exe -m high_level_training.evaluation.audit_dataset `
  --events D:\datasets\virat\dataset\high_level_events.csv `
  --snapshots D:\datasets\virat\dataset\high_level_snapshots_labeled.csv `
  --report D:\datasets\virat\dataset\audit_report.json
```

Báo cáo kiểm tra:

- Schema và kiểu số của đặc trưng.
- `event_id` trùng hoặc snapshot không có event.
- Confidence ngoài `[0, 1]`.
- Giá trị âm hoặc count không phải số nguyên.
- Số camera/video.
- Phân bố `LOW/MEDIUM/HIGH` và dữ liệu chưa gán nhãn.
- Điều kiện tối thiểu để bắt đầu train.

## 9. Train Random Forest

```powershell
D:\venvs\intrusion\Scripts\python.exe -m high_level_training.modeling.train_random_forest `
  --snapshots D:\datasets\virat\dataset\high_level_snapshots_labeled.csv `
  --output D:\datasets\virat\trained-model
```

Script chia tập theo `camera_id`; snapshot cùng sự kiện không bị lọt sang split khác. Chỉ nên
train sau khi có ít nhất ba camera và mỗi lớp có đủ dữ liệu để đánh giá.

## 10. Mốc kiểm tra pilot

Chỉ mở rộng sang toàn bộ VIRAT khi 10 video đầu đạt các điều kiện:

- Polygon đúng với từng góc camera.
- YOLO phát hiện người đủ ổn định.
- ByteTrack không đổi ID liên tục.
- FSM không tạo nhiều event cho một lần đứng trong vùng.
- Đặc trưng tốc độ, thời gian, độ sâu và count không bất thường.
- Evidence đủ để xác nhận nhãn.
