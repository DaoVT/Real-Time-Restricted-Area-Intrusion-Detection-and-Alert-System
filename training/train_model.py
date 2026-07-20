import os
import shutil
import yaml
import random
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download
from ultralytics import YOLO

# ==========================================
# CẤU HÌNH THƯ MỤC ĐẦU RA
# ==========================================
mega_dir = "/kaggle/working/mega_dataset"
for split in ['train', 'val', 'test']:
    os.makedirs(f"{mega_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{mega_dir}/labels/{split}", exist_ok=True)

def copy_file(src_img, src_lbl, target_split, new_name):
    shutil.copy(src_img, f"{mega_dir}/images/{target_split}/{new_name}.jpg")
    if os.path.exists(src_lbl):
        shutil.copy(src_lbl, f"{mega_dir}/labels/{target_split}/{new_name}.txt")

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN CÁC BỘ DATASET TRÊN KAGGLE
# ==========================================
print("Đang cấu hình đường dẫn dataset...", flush=True)

user_llvip_input = "/kaggle/input/llvip-dataset"
if not os.path.exists(user_llvip_input):
    user_llvip_input = "/kaggle/input/datasets/minhnguyncng19072005/llvip-dataset"

user_kaggle_input = "/kaggle/input/virat-dataset"
if not os.path.exists(user_kaggle_input):
    user_kaggle_input = "/kaggle/input/datasets/deadgirlwalking/virat-dataset"

# 1. Xác định llvip_base thực tế (hỗ trợ cả trường hợp bị lồng thêm 1 thư mục con)
llvip_base = user_llvip_input
if os.path.exists(user_llvip_input):
    if not os.path.exists(f"{llvip_base}/visible") or not os.path.exists(f"{llvip_base}/Annotations"):
        # Thử quét tìm thư mục con LLVIP bên trong
        for sub in os.listdir(user_llvip_input):
            sub_path = os.path.join(user_llvip_input, sub)
            if os.path.isdir(sub_path) and os.path.exists(f"{sub_path}/visible") and os.path.exists(f"{sub_path}/Annotations"):
                llvip_base = sub_path
                break

# 2. Xác định kaggle_base thực tế
kaggle_base = user_kaggle_input

if os.path.exists(llvip_base) and (os.path.exists(f"{llvip_base}/visible") or os.path.exists(f"{llvip_base}/Annotations")):
    print(f"-> Đã tìm thấy LLVIP tại: {llvip_base}", flush=True)
else:
    raise FileNotFoundError("❌ ERROR: Không tìm thấy thư mục LLVIP Dataset! Hãy chắc chắn đường dẫn tồn tại và chứa thư mục 'visible'/'Annotations'.")

if os.path.exists(kaggle_base):
    print(f"-> Đã tìm thấy Kaggle Public Dataset tại: {kaggle_base}", flush=True)
    # Đếm thử số lượng ảnh trong các thư mục 'images' tìm thấy
    total_images_found = 0
    for root, dirs, files in os.walk(kaggle_base):
        if os.path.basename(root) == 'images':
            jpg_count = len([f for f in files if f.endswith('.jpg')])
            print(f"   + Tìm thấy thư mục ảnh: {root.replace(kaggle_base, '')} (chứa {jpg_count} file)", flush=True)
            total_images_found += jpg_count
    print(f"   => Tổng cộng tìm thấy {total_images_found} ảnh trong Kaggle Public Dataset.", flush=True)
else:
    raise FileNotFoundError(f"❌ ERROR: Không tìm thấy thư mục Kaggle Public Dataset tại {user_kaggle_input}! Vui lòng kiểm tra lại đường dẫn.")

# ==========================================
# 1. LLVIP DATASET
# Đã có sẵn Train/Test. Đọc từ file XML. Tách Train thành 90% Train / 10% Val theo Scene
# ==========================================
print("\n1. Xử lý LLVIP Dataset...", flush=True)

def process_llvip_split(split_name, is_test=False):
    img_dir = f"{llvip_base}/visible/{split_name}"
    anno_dir = f"{llvip_base}/Annotations"
    if not os.path.exists(img_dir): 
        print(f"⚠️ Cảnh báo: Không tồn tại thư mục ảnh: {img_dir}", flush=True)
        return
    
    imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    # Gom nhóm theo scene (2 chữ số đầu của tên file) để tránh leakage
    scene_groups = defaultdict(list)
    for img_name in imgs:
        scene_id = img_name[:2]
        scene_groups[scene_id].append(img_name)
        
    scenes = list(scene_groups.keys())
    if not is_test:
        random.shuffle(scenes)
        train_cut = int(len(scenes) * 0.9)
        train_scenes = set(scenes[:train_cut])
    
    copied_count = 0
    for scene_id in scenes:
        if is_test:
            target = 'test'
        else:
            target = 'train' if scene_id in train_scenes else 'val'
            
        for img_name in tqdm(scene_groups[scene_id], desc=f"LLVIP {split_name} - Scene {scene_id}"):
            base_name = os.path.splitext(img_name)[0]
            xml_path = os.path.join(anno_dir, f"{base_name}.xml")
            
            if os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    size = root.find('size')
                    img_w, img_h = float(size.find('width').text), float(size.find('height').text)
                    
                    labels = ""
                    for obj in root.findall('object'):
                        if obj.find('name').text.lower() == 'person':
                            bndbox = obj.find('bndbox')
                            xmin, ymin = float(bndbox.find('xmin').text), float(bndbox.find('ymin').text)
                            xmax, ymax = float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)
                            
                            cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                            w, h = xmax - xmin, ymax - ymin
                            
                            nx, ny, nw, nh = max(0, min(1, cx/img_w)), max(0, min(1, cy/img_h)), max(0, min(1, w/img_w)), max(0, min(1, h/img_h))
                            labels += f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n"
                            
                    if labels:
                        new_name = f"LLVIP_{base_name}"
                        shutil.copy(os.path.join(img_dir, img_name), f"{mega_dir}/images/{target}/{new_name}.jpg")
                        with open(f"{mega_dir}/labels/{target}/{new_name}.txt", "w") as f:
                            f.write(labels)
                        copied_count += 1
                except Exception:
                    pass
    print(f"-> Đã copy {copied_count} ảnh từ LLVIP {split_name} vào tập {target}.", flush=True)

process_llvip_split('train', is_test=False)
process_llvip_split('test', is_test=True)

# ==========================================
# 2. VIRAT + MultiviewX + CVLab (Kaggle Public)
# Gom toàn bộ từ các thư mục gốc và chia theo Scene/Video (70% Train, 15% Val, 15% Test)
# ==========================================
print("\n2. Xử lý Kaggle Public Dataset...", flush=True)

video_groups = defaultdict(list)

# Quét đệ quy tìm tất cả thư mục 'images' của dataset public để không bỏ sót dữ liệu
for root, dirs, files in os.walk(kaggle_base):
    if os.path.basename(root) == 'images':
        lbl_dir = root.replace('/images', '/labels').replace('\\images', '\\labels')
        for file in files:
            if not file.endswith('.jpg'): continue
            # Gom theo prefix của video để chia theo sequence tránh leakage
            prefix = "_".join(file.split('_')[:-1]) 
            video_groups[prefix].append({
                'img_path': os.path.join(root, file),
                'lbl_path': os.path.join(lbl_dir, os.path.splitext(file)[0] + '.txt'),
                'img_name': file
            })

copied_kaggle_count = 0
if video_groups:
    video_prefixes = list(video_groups.keys())
    random.shuffle(video_prefixes)
    
    train_idx = int(len(video_prefixes) * 0.7)
    val_idx = int(len(video_prefixes) * 0.85)
    
    for i, prefix in enumerate(tqdm(video_prefixes, desc="Kaggle Dataset (by Scene)")):
        if i < train_idx: target = 'train'
        elif i < val_idx: target = 'val'
        else: target = 'test'
        
        for item in video_groups[prefix]:
            new_name = f"KAGGLE_{os.path.splitext(item['img_name'])[0]}"
            copy_file(item['img_path'], item['lbl_path'], target, new_name)
            copied_kaggle_count += 1
            
print(f"-> Đã copy {copied_kaggle_count} ảnh từ Kaggle Public Dataset.", flush=True)

# ==========================================
# 3. HUGGING FACE DATASET (MOT20)
# Tải và xử lý ảnh từ MOT20
# ==========================================
print("\n3. Xử lý MOT20 Dataset...", flush=True)

# Tìm HF_TOKEN trong Kaggle Secrets
hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    if hf_token:
        print("Đã tải HF_TOKEN từ Secrets thành công!", flush=True)
except Exception:
    print("Tải ẩn danh không token (Unauthenticated)...", flush=True)

# Tải và giải nén MOT20 từ HuggingFace Hub với log in chi tiết tiến trình tải
print("Đang kết nối Hugging Face API để lấy danh sách file...", flush=True)
api = HfApi(token=hf_token)
try:
    all_files = api.list_repo_files(repo_id="Lekim89/MOT20", repo_type="dataset")
except Exception as e:
    print(f"❌ Lỗi kết nối Hugging Face: {e}", flush=True)
    raise e

# Lọc các file cần tải
files_to_download = []
for f in all_files:
    if any(seq in f for seq in ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']):
        if (f.endswith('.jpg') and 'img1' in f) or (f.endswith('.txt') and 'gt' in f):
            files_to_download.append(f)

total_files = len(files_to_download)
print(f"🔍 Đã quét xong. Tổng số file cần tải: {total_files}", flush=True)

cache_dir = "/kaggle/working/mot20_cache"
os.makedirs(cache_dir, exist_ok=True)

downloaded_count = 0

def download_single_file(file_path):
    global downloaded_count
    max_retries = 5
    backoff = 2
    for attempt in range(max_retries):
        try:
            local_path = hf_hub_download(
                repo_id="Lekim89/MOT20",
                filename=file_path,
                repo_type="dataset",
                token=hf_token,
                local_dir=cache_dir,
                local_dir_use_symlinks=False
            )
            downloaded_count += 1
            if downloaded_count % 50 == 0 or downloaded_count == total_files:
                print(f"[Tiến độ] Đã tải {downloaded_count}/{total_files} file ({downloaded_count/total_files*100:.1f}%)", flush=True)
            return local_path
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ Lỗi tải file {file_path}: {e}", flush=True)
                raise e
            time.sleep(backoff ** attempt + 1)

print("🚀 Bắt đầu tải đa luồng (max_workers=4) từ Hugging Face...", flush=True)
start_time = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(download_single_file, f) for f in files_to_download]
    for future in as_completed(futures):
        try:
            future.result()
        except Exception:
            pass

print(f"✅ Hoàn tất tải MOT20! Tổng thời gian: {time.time() - start_time:.1f} giây.", flush=True)
mot20_base = f"{cache_dir}/train"
mot20_seqs = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']

for seq in mot20_seqs:
    target = 'val' if seq == 'MOT20-01' else 'train'
    seq_dir = f"{mot20_base}/{seq}"
    if not os.path.exists(seq_dir): continue
        
    img_dir = f"{seq_dir}/img1"
    gt_file = f"{seq_dir}/gt/gt.txt"
    
    sample_img = os.path.join(img_dir, os.listdir(img_dir)[0])
    with Image.open(sample_img) as img:
        img_w, img_h = img.size
        
    frame_labels = defaultdict(list)
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9: continue # Phải đủ 9 trường mới xử lý
            
            conf = int(parts[6])
            if conf == 0: continue # Box đánh dấu ignore
            
            cls_id = int(parts[7])
            if cls_id not in [1, 2, 7]: continue # Chỉ lấy người
            
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            
            x_center = (x + w/2) / img_w
            y_center = (y + h/2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            
            frame_labels[int(parts[0])].append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
    for img_name in tqdm(os.listdir(img_dir), desc=f"MOT20 {seq}"):
        if not img_name.endswith('.jpg'): continue
        frame_id = int(os.path.splitext(img_name)[0])
        
        new_name = f"MOT20_{seq}_{frame_id}"
        shutil.copy(os.path.join(img_dir, img_name), f"{mega_dir}/images/{target}/{new_name}.jpg")
        
        if frame_id in frame_labels:
            with open(f"{mega_dir}/labels/{target}/{new_name}.txt", "w") as f:
                f.write("\n".join(frame_labels[frame_id]))

print("4. Tạo data.yaml...", flush=True)
yaml_content = {
    'path': mega_dir, 
    'train': "images/train", 
    'val': "images/val", 
    'test': "images/test",
    'names': {0: 'person'}
}
with open(f"{mega_dir}/data.yaml", "w") as f:
    yaml.dump(yaml_content, f, sort_keys=False)
    
print("\n=== THỐNG KÊ SỐ LƯỢNG ẢNH SAU KHI GỘP ===", flush=True)
for split in ['train', 'val', 'test']:
    img_count = len(os.listdir(f"{mega_dir}/images/{split}"))
    lbl_count = len(os.listdir(f"{mega_dir}/labels/{split}"))
    print(f"Tập {split.upper()}: {img_count} ảnh, {lbl_count} file nhãn", flush=True)
    
    # Chặn lỗi: Nếu có bất kỳ tập nào không có ảnh, báo lỗi dừng ngay lập tức
    assert img_count > 0, f"❌ ERROR: Tập {split.upper()} trống rỗng! Đã có lỗi xảy ra khi gộp hoặc chia dữ liệu."
    
    # Cảnh báo nếu số nhãn quá ít so với số lượng ảnh
    if lbl_count == 0:
        print(f"⚠️ Cảnh báo: Tập {split.upper()} không có file nhãn nào!", flush=True)
    elif lbl_count < img_count * 0.5:
        print(f"⚠️ Cảnh báo: Tập {split.upper()} có số nhãn ({lbl_count}) ít hơn nhiều so với số ảnh ({img_count}).", flush=True)

print("\n✅ KIỂM TRA ĐỘ TOÀN VẸN: HỢP LỆ! Dữ liệu đã chia tách chuẩn chỉnh.", flush=True)

# ==========================================
# 4. HUẤN LUYỆN DỰ TIẾN HÀNH (YOLOv8n)
# ==========================================
print("\n=== BẮT ĐẦU HUẤN LUYỆN YOLOv8n ===", flush=True)
model = YOLO('yolov8n.pt')

# Chạy train chống overfitting (patience=15, verbose=False để hạn chế phình dung lượng log)
results = model.train(
    data=f"{mega_dir}/data.yaml",
    epochs=100, 
    imgsz=640,
    batch=16,
    device=[0, 1],  # Chạy song song cả 2 GPU T4 của Kaggle
    patience=15,    # Dừng sớm nếu không cải thiện sau 15 epoch
    save_period=5,  # Lưu checkpoint mỗi 5 epoch
    project='/kaggle/working/runs',
    name='mega_model',
    verbose=False   # Giúp file notebook không bị phình dung lượng log
)

print("\n✅ HOÀN TẤT HUẤN LUYỆN DỰ ÁN!", flush=True)
