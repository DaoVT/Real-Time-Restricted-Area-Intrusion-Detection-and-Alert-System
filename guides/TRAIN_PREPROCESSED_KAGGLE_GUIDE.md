# Hướng Dẫn Huấn Luyện (Training) YOLO Bằng Cách Tái Sử Dụng Dataset Cũ

Tài liệu này hướng dẫn bạn cách **tái sử dụng 100% tập dữ liệu `mega_dataset` đã chia sẵn** từ lần chạy Kaggle trước. Phương pháp này đảm bảo tính công bằng tuyệt đối: cả 2 lần train (Ảnh gốc và Ảnh Filter) đều dùng chung một bộ ảnh train/val/test giống hệt nhau về mặt vật lý.

---

## Bước 1: Chuẩn bị trên Kaggle
1. Tạo một Notebook Kaggle mới (hoặc dùng lại notebook cũ nhưng xóa hết các ô code đi).
2. Đảm bảo Notebook đang bật **GPU T4 x2** và **Internet on**.
3. Bấm vào nút **Add Data** (ở cột bên phải) -> Chọn tab **Your Work** -> Tìm và thêm Output của Notebook cũ (nơi chứa thư mục `mega_dataset` mà bạn vừa chụp ảnh).

> *Lưu ý: Sau khi Add xong, dataset cũ của bạn sẽ nằm ở một đường dẫn kiểu như: `/kaggle/input/notebook560d537205/mega_dataset` (tên thư mục có thể khác tùy thuộc vào tên notebook của bạn).*

---

## Bước 2: Các Ô Code Cần Chạy

### Ô số 1: Cài đặt thư viện
```python
!pip install ultralytics Pillow tqdm opencv-python-headless
```

### Ô số 2: Khai báo đường dẫn Data cũ
Chạy đoạn code này để máy tự động tìm thư mục `mega_dataset` mà bạn vừa Add vào:
```python
import os
import shutil

INPUT_DATA_PATH = ""
for root, dirs, files in os.walk('/kaggle/input'):
    if 'mega_dataset' in dirs:
        INPUT_DATA_PATH = os.path.join(root, 'mega_dataset')
        break

if INPUT_DATA_PATH:
    print(f"Đã tìm thấy Dataset cũ tại: {INPUT_DATA_PATH}")
else:
    print("CHƯA TÌM THẤY! Hãy chắc chắn bạn đã Add Data thành công.")
```

### Ô số 3: Copy Data và Áp dụng Bộ lọc (Nếu cần)
Để YOLO có thể chạy và chúng ta có quyền sửa ảnh (đè bộ lọc lên), ta phải copy data sang `/kaggle/working`.
Trong ô code này có biến `ENABLE_FILTERS`. 
- Nếu muốn chạy **Baseline (Ảnh gốc)**: Để `False`.
- Nếu muốn chạy **Filtered (Đã qua xử lý)**: Để `True`.

```python
import cv2
import numpy as np
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# CÔNG TẮC BẬT/TẮT BỘ LỌC
# ==========================================
ENABLE_FILTERS = True  # Để True nếu muốn chạy bản Tiền xử lý (Filtered) trước

# Thông số bộ lọc
VERY_DARK_THRESHOLD, DARK_THRESHOLD, BLUR_THRESHOLD = 40, 75, 80
CLAHE_CLIP, UNSHARP_STRENGTH, GAMMA_VALUE = 1.8, 0.6, 1.8

def detect_best_filter(frame):
    if frame is None or frame.size == 0: return "none"
    h, w = frame.shape[:2]
    scale = min(320 / w, 180 / h, 1.0)
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
    
    mean_b = np.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    blur_m = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    
    if mean_b < VERY_DARK_THRESHOLD: return "gamma_unsharp" if blur_m < BLUR_THRESHOLD else "gamma_correction"
    elif mean_b < DARK_THRESHOLD: return "clahe_unsharp" if blur_m < BLUR_THRESHOLD else "clahe"
    elif blur_m < BLUR_THRESHOLD: return "unsharp_mask"
    return "none"

def apply_filters(frame, f_type):
    if f_type == "none": return frame
    
    def apply_clahe(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        
    def apply_gamma(img):
        inv_gamma = 1.0 / GAMMA_VALUE
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.fastNlMeansDenoisingColored(cv2.LUT(img, table), None, 5, 5, 7, 21)
        
    def apply_unsharp(img):
        blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
        return cv2.addWeighted(img, 1.0 + UNSHARP_STRENGTH, blurred, -UNSHARP_STRENGTH, 0)
        
    if f_type == "clahe": return apply_clahe(frame)
    elif f_type == "gamma_correction": return apply_gamma(frame)
    elif f_type == "unsharp_mask": return apply_unsharp(frame)
    elif f_type == "clahe_unsharp": return apply_unsharp(apply_clahe(frame))
    elif f_type == "gamma_unsharp": return apply_unsharp(apply_gamma(frame))
    return frame

def process_single_image(img_path):
    frame = cv2.imread(img_path)
    if frame is None: return
    f_type = detect_best_filter(frame)
    processed_frame = apply_filters(frame, f_type)
    cv2.imwrite(img_path, processed_frame) # Ghi đè trực tiếp

# 1. Copy Data sang working
WORKING_DIR = "/kaggle/working/mega_dataset"
if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR) # Xóa cũ nếu chạy lại
print("Đang copy data sang /kaggle/working (Mất khoảng 1-2 phút)...")
shutil.copytree(INPUT_DATA_PATH, WORKING_DIR)

# Sửa lại file yaml cho chuẩn đường dẫn Kaggle
with open(f"{WORKING_DIR}/data.yaml", "r") as f:
    yaml_data = yaml.safe_load(f)
yaml_data['path'] = WORKING_DIR
with open(f"{WORKING_DIR}/data.yaml", "w") as f:
    yaml.dump(yaml_data, f, sort_keys=False)

# 2. Áp dụng bộ lọc (Nếu bật)
if ENABLE_FILTERS:
    print("ĐANG ÁP DỤNG BỘ LỌC LÊN HÀNG NGÀN BỨC ẢNH (Hãy kiên nhẫn)...")
    img_paths = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(WORKING_DIR, 'images', split)
        if os.path.exists(split_dir):
            for f in os.listdir(split_dir):
                if f.endswith('.jpg'): img_paths.append(os.path.join(split_dir, f))
                
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_single_image, img_paths), total=len(img_paths), desc="Filtering Images"))
    print("Hoàn tất bộ lọc!")
else:
    print("Chế độ Baseline: Giữ nguyên ảnh gốc không lọc.")
```

### Ô số 4: Huấn luyện mô hình (Training)
```python
import torch
from ultralytics import YOLO

# Khóa chặt các biến số ngẫu nhiên của PyTorch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Khởi tạo mô hình cơ bản (YOLOv8 Nano) như gốc
model = YOLO('yolov8n.pt') 

# Tự động đặt tên thư mục
model_name = 'mega_model_filtered' if ENABLE_FILTERS else 'mega_model_baseline'

# Bắt đầu quá trình huấn luyện 100 epochs
results = model.train(
    data='/kaggle/working/mega_dataset/data.yaml',
    epochs=100,      
    imgsz=640,
    batch=16,
    device=[0, 1],  # Sử dụng 2x T4 GPU
    patience=15,    
    save_period=5,  
    project='/kaggle/working/runs',
    name=model_name,
    exist_ok=True,       # Ghi đè thư mục cũ để đường dẫn tải model luôn đúng
    seed=42              # Khóa seed của YOLO
)
```

### Ô số 5: Đánh giá & Tải Model
```python
from ultralytics import YOLO
from IPython.display import FileLink, display

# Tự động tìm đúng thư mục
model_name = 'mega_model_filtered' if ENABLE_FILTERS else 'mega_model_baseline'
model_path = f'/kaggle/working/runs/{model_name}/weights/best.pt'

best_model = YOLO(model_path)

print(f"=== TIẾN HÀNH ĐÁNH GIÁ ({model_name}) ===")
metrics = best_model.val(data='/kaggle/working/mega_dataset/data.yaml', split='test')

print(f"mAP50: {metrics.box.map50:.3f}")
print(f"Precision: {metrics.box.p[0]:.3f}")
print(f"Recall: {metrics.box.r[0]:.3f}")

print("\n=== TẢI MÔ HÌNH VỀ MÁY ===")
display(FileLink(model_path))
```

---

## Tóm tắt quy trình làm việc
1. Chạy 5 ô với `ENABLE_FILTERS = True` để lấy **Filtered Model**. Tải file `best.pt` về máy.
2. Sửa lại `ENABLE_FILTERS = False`, bấm chạy lại Ô 3, 4, 5 để lấy **Baseline Model**. Tải file mới về và đối chiếu.
