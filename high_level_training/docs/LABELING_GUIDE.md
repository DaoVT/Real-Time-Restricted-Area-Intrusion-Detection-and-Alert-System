# Hướng dẫn gán nhãn rủi ro

Model mức cao chỉ đánh giá sự kiện đã được FSM xác nhận `INTRUDING`. Trường hợp không xâm nhập không được gán thành `LOW`.

## Chọn mục tiêu gán nhãn

Không trộn hai cách sau trong cùng một dataset:

- **Phân loại mức cuối của sự kiện:** xem toàn bộ sự kiện, gán một nhãn trong
  `high_level_events.csv`, sau đó dùng `propagate_event_labels.py`. Đây là baseline dễ thực hiện,
  nhưng snapshot đầu cũng mang nhãn cuối của sự kiện.
- **Đánh giá rủi ro realtime:** gán riêng từng dòng `high_level_snapshots.csv`. Cách này tốn công
  hơn nhưng phản ánh đúng việc mức nguy hiểm tăng hoặc giảm trong khi sự kiện diễn ra.

Nhãn trong `high_level_snapshots.csv` biểu diễn mức nguy hiểm tại đúng thời điểm của snapshot:

- `LOW`: mới vào, ở gần mép vùng, di chuyển chậm, chưa có hành vi đáng chú ý.
- `MEDIUM`: đi tương đối sâu, ở lâu hơn, đứng lại hoặc đổi hướng rõ ràng.
- `HIGH`: chạy nhanh, tiến sâu, đứng lâu tại vị trí nhạy cảm, quay lại hoặc có nhiều người cùng xâm nhập.

Quy tắc gán nhãn:

1. Xem evidence hoặc video quanh thời điểm `video_time_sec`.
2. Điền một trong ba nhãn `LOW`, `MEDIUM`, `HIGH`.
3. Ghi lý do ngắn vào `annotator_note`.
4. Các tình huống tương tự phải được gán nhãn nhất quán.
5. Snapshot đầu và cuối cùng một sự kiện có thể mang nhãn khác nhau vì rủi ro thay đổi theo thời gian.

## Gán nhãn sơ bộ tự động

`auto_label_dataset.py` tạo **weak labels** để khởi tạo baseline. Script không sửa CSV gốc,
khôi phục `camera_id` của snapshot từ event tương ứng và ghi lý do chấm điểm vào
`annotator_note`.

Mỗi đặc trưng vượt ngưỡng vừa được cộng 1 điểm, vượt ngưỡng cao được cộng 2 điểm:

| Đặc trưng | Ngưỡng vừa | Ngưỡng cao |
| --- | ---: | ---: |
| Thời gian trong vùng | 8 giây | 30 giây |
| Tốc độ trung bình chuẩn hóa | 0.015 | 0.030 |
| Tốc độ cực đại chuẩn hóa | 0.050 | 0.100 |
| Độ sâu trong vùng chuẩn hóa | 0.050 | 0.100 |
| Thời gian đứng yên | 5 giây | 20 giây |
| Số lần đổi hướng | 3 | 8 |
| Số người cực đại trong vùng | 3 | 5 |
| Số lần tái xâm nhập | - | 1 |

- `LOW`: tổng điểm từ 0 đến 2.
- `MEDIUM`: tổng điểm từ 3 đến 6.
- `HIGH`: tổng điểm từ 7 trở lên.

Các ngưỡng này được đối chiếu với phân bố dataset v6 nhưng vẫn là quy ước nghiệp vụ.
Không nên coi weak labels là ground truth. Cần xem lại một mẫu ngẫu nhiên ở từng nhãn,
ưu tiên toàn bộ mẫu `HIGH`, rồi sửa những nhãn không phù hợp trước khi đánh giá model.

### Chính sách hành vi v2

Policy `behavioral-v2` dùng quy tắc nghiệp vụ đã được duyệt thay cho phép cộng điểm ngang nhau:

- Dấu hiệu mạnh: ở trong vùng từ 20 giây, đứng yên từ 15 giây, tái xâm nhập,
  có ít nhất 5 người, tốc độ cực đại từ 0,1 hoặc độ sâu từ 0,1.
- `HIGH`: có ít nhất hai dấu hiệu mạnh và phải có một dấu hiệu bối cảnh gồm ở lâu,
  đứng lâu, tái xâm nhập hoặc đông người.
- `MEDIUM`: chưa đủ điều kiện HIGH nhưng có ít nhất một dấu hiệu mức vừa.
- `LOW`: chưa có dấu hiệu mức vừa.

Điều kiện bối cảnh ngăn trường hợp chỉ chạy nhanh và đi sâu trong thời gian rất ngắn
bị đẩy thẳng lên HIGH.

```powershell
python -m high_level_training.labeling.auto_label_dataset `
  --events D:\datasets\high-level-combined-v6\dataset\high_level_events.csv `
  --snapshots D:\datasets\high-level-combined-v6\dataset\high_level_snapshots.csv `
  --output-dir D:\datasets\high-level-combined-v8-behavioral\dataset `
  --policy behavioral-v2
```

```powershell
python -m high_level_training.labeling.auto_label_dataset `
  --events D:\datasets\high-level-combined-v6\dataset\high_level_events.csv `
  --snapshots D:\datasets\high-level-combined-v6\dataset\high_level_snapshots.csv `
  --output-dir D:\datasets\high-level-combined-v6-weak-labeled\dataset
```

Ví dụ:

```csv
event_id,video_time_sec,label,annotator_note
event_001,2.0,LOW,"Người mới bước vào sát mép vùng"
event_001,5.0,MEDIUM,"Người đi sâu hơn và bắt đầu đứng lại"
event_001,8.0,HIGH,"Người đứng lâu tại vị trí nhạy cảm"
```
