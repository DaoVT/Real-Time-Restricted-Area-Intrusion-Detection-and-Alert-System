# Training mức cao

Folder này chỉ phục vụ model đánh giá rủi ro `LOW/MEDIUM/HIGH`. Pipeline YOLO mức trung vẫn nằm trong `training/` và không bị thay đổi.

## Cấu trúc thư mục

```text
high_level_training/
├── data/        # Thu thập video, tạo manifest và gộp CSV
├── labeling/    # Tạo nhãn tự động, lan truyền nhãn và áp dụng nhãn đã duyệt
├── modeling/    # Huấn luyện model Random Forest
├── evaluation/  # Kiểm tra dataset, tạo biểu đồ và phân tích dự đoán sai
├── docs/        # Hướng dẫn gán nhãn và quy trình xử lý dataset
└── examples/    # File manifest mẫu
```

Mỗi file Python được chạy từ thư mục gốc project bằng tên module đầy đủ, ví dụ:

```powershell
python -m high_level_training.data.collect_dataset --help
python -m high_level_training.modeling.train_random_forest --help
```

## Dữ liệu được tạo

Mỗi job xử lý video tạo thêm:

- `high_level_events.csv`: một dòng tổng hợp khi một sự kiện xâm nhập kết thúc.
- `high_level_snapshots.csv`: một dòng đặc trưng sau mỗi 0,5 giây video trong lúc sự kiện diễn ra.

Các cột `label` và `annotator_note` được để trống. Người gán nhãn cần xem lại evidence/video trước khi điền `LOW`, `MEDIUM` hoặc `HIGH`.

## Gộp nhiều job

Nên lưu dataset lớn ngoài repository, ví dụ `D:\intrusion-datasets`:

```powershell
python -m high_level_training.data.prepare_dataset --source backend/output --destination D:\intrusion-datasets
```

Lệnh chỉ gộp CSV, không train YOLO và không sửa `best.pt`.

## Thu thập nhiều video

Tạo manifest dựa trên `examples/manifest.example.csv`. `camera_id` là mã góc quay tự đặt; các đoạn cùng nguồn/góc quay dùng cùng mã.

```powershell
python -m high_level_training.data.collect_dataset `
  --manifest high_level_training\examples\manifest.example.csv `
  --output-root backend\output\collection `
  --model yolov8n.pt
```

Output chỉ được coi là hoàn tất khi có `.collection_complete.json`. Folder có file nhưng thiếu
marker được xem là lần chạy dở và không bị ghi đè tự động. Công cụ tạo detector mới cho từng
video nhằm cô lập trạng thái ByteTrack.

Đối với VIRAT Ground, xem quy trình đầy đủ tại `docs/VIRAT_WORKFLOW.md`. Công cụ tạo manifest:

```powershell
python -m high_level_training.data.prepare_virat `
  --raw-root D:\datasets\virat\raw `
  --zones-root D:\datasets\virat\zones `
  --output D:\datasets\virat\manifests\pilot.csv
```

Kiểm tra manifest mà không tải YOLO:

```powershell
python -m high_level_training.data.collect_dataset `
  --manifest D:\datasets\virat\manifests\pilot.csv `
  --output-root D:\datasets\virat\outputs `
  --dry-run `
  --limit 10
```

Đối với MOT20, dữ liệu gốc là chuỗi ảnh. Tạo một file zone theo từng sequence,
ví dụ `zones\MOT20-01.json`, rồi chuyển ảnh thành video và tạo manifest:

```powershell
python -m high_level_training.data.prepare_mot `
  --raw-root D:\datasets\mot20\raw\selected `
  --zones-root D:\datasets\mot20\zones `
  --videos-root D:\datasets\mot20\videos `
  --output D:\datasets\mot20\manifests\pilot.csv `
  --sequence MOT20-01
```

Công cụ kiểm tra đủ số frame theo `seqinfo.ini`, giữ nguyên FPS và kích thước ảnh,
sau đó mở lại MP4 để xác nhận file có thể đọc được. Mỗi sequence được xem là một
camera độc lập khi chia train/validation/test.

## Gán nhãn và kiểm tra

Có hai cách:

- Gán trực tiếp từng snapshot nếu mức rủi ro thay đổi theo thời gian.
- Gán một nhãn cuối trong event rồi dùng `propagate_event_labels` để tạo baseline.

Luôn tạo file snapshot mới, không ghi đè file gốc:

```powershell
python -m high_level_training.labeling.propagate_event_labels `
  --events D:\intrusion-datasets\high_level_events.csv `
  --snapshots D:\intrusion-datasets\high_level_snapshots.csv `
  --output D:\intrusion-datasets\high_level_snapshots_labeled.csv
```

Kiểm tra dataset trước khi train:

```powershell
python -m high_level_training.evaluation.audit_dataset `
  --events D:\intrusion-datasets\high_level_events.csv `
  --snapshots D:\intrusion-datasets\high_level_snapshots_labeled.csv `
  --report D:\intrusion-datasets\audit_report.json
```

## Bước tiếp theo

1. Chạy nhiều video đa dạng để tạo event và snapshot.
2. Gán nhãn theo `docs/LABELING_GUIDE.md`.
3. Kiểm tra phân bố lớp và chia dataset theo camera/video.
4. Chạy audit và chỉ train Random Forest khi dữ liệu đủ tin cậy.

## Train Random Forest

Cài dependency riêng cho mức cao:

```powershell
python -m pip install -r high_level_training\requirements.txt
```

Sau khi `high_level_snapshots.csv` đã được gán nhãn và có ít nhất ba camera/video độc lập:

```powershell
python -m high_level_training.modeling.train_random_forest `
  --snapshots D:\intrusion-datasets\high_level_snapshots.csv `
  --output D:\intrusion-datasets\trained-model
```

Kết quả gồm `risk_model.joblib`, `metrics.json`, `confusion_matrix.csv`, `feature_importance.csv` và `split_assignments.csv`.

Script tự bỏ qua dòng chưa gán nhãn, nhưng sẽ dừng nếu gặp nhãn ngoài `LOW`, `MEDIUM`, `HIGH`, dữ liệu không phải số hoặc có ít hơn ba nguồn độc lập.

## Tạo biểu đồ đánh giá

Sau khi train, tạo báo cáo từ model và phép chia dữ liệu đã lưu:

```powershell
python -m high_level_training.evaluation.generate_evaluation_report `
  --snapshots D:\datasets\high-level-combined-v8-behavioral\dataset\high_level_snapshots.csv `
  --model D:\datasets\high-level-model-v5\risk_model.joblib `
  --assignments D:\datasets\high-level-model-v5\split_assignments.csv `
  --output D:\datasets\high-level-model-v5\evaluation
```

Công cụ chỉ dự đoán trên tập `test`, không train lại hoặc ghi đè model. Kết quả gồm:

- `confusion_matrix.png`: số lượng nhãn thật và nhãn dự đoán.
- `feature_importance.png`: mức đóng góp tương đối của từng thuộc tính.
- `precision_recall_curve.png`: Precision–Recall cho từng mức rủi ro.
- `roc_curve.png`: ROC–AUC theo cách one-vs-rest.
- `confidence_threshold_curve.png`: Precision, Recall và F1 khi thay đổi ngưỡng xác suất.
- `class_distribution.png`: phân bố nhãn trong train, validation và test.
- `test_predictions.csv`: dự đoán và xác suất của từng snapshot test.
- `evaluation_summary.json`: số liệu dùng để tái lập báo cáo.
