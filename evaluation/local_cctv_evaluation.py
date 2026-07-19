import os
import sys
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

# Tự động chuyển thư mục làm việc về thư mục gốc dự án
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
sys.path.append(str(project_root))

def main():
    print("=== BƯỚC 1: LỌC BỎ FLYCAM, CHỈ GIỮ LẠI CAMERA AN NINH ===")
    source_img_dir = 'Mega_Dataset_Final/images/test'
    source_lbl_dir = 'Mega_Dataset_Final/labels/test'

    cctv_img_dir = 'cctv_test/images'
    cctv_lbl_dir = 'cctv_test/labels'
    os.makedirs(cctv_img_dir, exist_ok=True)
    os.makedirs(cctv_lbl_dir, exist_ok=True)

    count = 0
    if os.path.exists(source_img_dir):
        for filename in os.listdir(source_img_dir):
            # CHỈ LOẠI BỎ VisDrone, giữ lại LLVIP, ExDark và WiderPerson
            if filename.startswith('llvip_') or filename.startswith('exdark_') or filename.startswith('wider_'):
                img_src = os.path.join(source_img_dir, filename)
                img_dst = os.path.join(cctv_img_dir, filename)
                if not os.path.exists(img_dst):
                    shutil.copy(img_src, img_dst)
                
                lbl_name = filename.rsplit('.', 1)[0] + '.txt'
                lbl_src = os.path.join(source_lbl_dir, lbl_name)
                lbl_dst = os.path.join(cctv_lbl_dir, lbl_name)
                if os.path.exists(lbl_src) and not os.path.exists(lbl_dst):
                    shutil.copy(lbl_src, lbl_dst)
                count += 1
        print(f"Đã chuẩn bị xong {count} bức ảnh CCTV vào đề thi mới!")
    else:
        print("Lỗi: Không tìm thấy thư mục Mega_Dataset_Final gốc. Dừng chương trình.")
        return

    print("\n=== BƯỚC 2: TẠO FILE YAML ===")
    cctv_yaml = {
        'path': os.path.abspath('cctv_test'),
        'train': 'images',
        'val': 'images',
        'nc': 1,
        'names': ['person']
    }
    yaml_path = 'cctv.yaml'
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(cctv_yaml, f)

    print("\n=== BƯỚC 3: BẮT ĐẦU CHẤM ĐIỂM CAMERA AN NINH ===")
    model_path = 'best.pt'
    
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file model {model_path} ở thư mục hiện tại!")
        return

    model = YOLO(model_path)
    # Dùng workers=0 trên Windows để tránh lỗi đa luồng
    metrics = model.val(data=yaml_path, split='val', workers=0)

    print(f"\n--- KẾT QUẢ ĐÁNH GIÁ TRÊN CAMERA AN NINH ---")
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"Precision: {metrics.box.p[0]:.3f}")
    print(f"Recall:    {metrics.box.r[0]:.3f}")

if __name__ == '__main__':
    main()
