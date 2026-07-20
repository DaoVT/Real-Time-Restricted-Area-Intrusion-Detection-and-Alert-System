import os
import shutil
import zipfile
import random
import sys
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

# Tự động chuyển thư mục làm việc về thư mục gốc dự án
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
sys.path.append(str(project_root))

# Configuration
OUTPUT_DIR = Path("Mega_Dataset_Final")
STRIDE_VISDRONE = 10 # Downsample VisDrone

# Output directories
SPLITS = ['train', 'val', 'test']

def setup_directories():
    if OUTPUT_DIR.exists():
        print(f"Removing old {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    
    for split in SPLITS:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print("Mega_Dataset directories created.")

# A global list to hold dataset items: {"img_data": bytes or None, "img_ext": str, "src_path": Path or str, "label_lines": list}
# We will split this list and then write to disk.
dataset_items = []

def process_exdark():
    print("Processing ExDark_YOLO...")
    exdark_dir = Path("ExDark_YOLO")
    if not exdark_dir.exists():
        print("ExDark_YOLO not found!")
        return
        
    items = []
    for split in ['train', 'val']:
        img_dir = exdark_dir / 'images' / split
        lbl_dir = exdark_dir / 'labels' / split
        if img_dir.exists():
            for img_path in img_dir.glob("*.jpg"):
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    with open(lbl_path, 'r') as f:
                        lines = [l.strip() for l in f.readlines()]
                    items.append({
                        "source": "random",
                        "prefix": "exdark_",
                        "img_path": img_path,
                        "label_lines": lines
                    })
    dataset_items.extend(items)
    print(f"Added {len(items)} ExDark items.")

def process_visdrone(zip_path):
    if not Path(zip_path).exists():
        print(f"VisDrone zip {zip_path} not found!")
        return
        
    print(f"Processing {zip_path}...")
    items = []
    
    split_source = "train"
    if "val" in zip_path: split_source = "val"
    elif "test" in zip_path: split_source = "test"
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = z.namelist()
        
        prefix = ""
        for name in namelist:
            if name.endswith("annotations/"):
                prefix = name.split("annotations/")[0]
                break
                
        anno_files = [f for f in namelist if f.startswith(f"{prefix}annotations/") and f.endswith(".txt")]
        
        for anno_file_path in anno_files:
            seq_name = Path(anno_file_path).stem
            try:
                anno_content = z.read(anno_file_path).decode('utf-8')
            except:
                continue
                
            seq_imgs_in_zip = [f for f in namelist if f.startswith(f"{prefix}sequences/{seq_name}/") and f.endswith(".jpg")]
            if not seq_imgs_in_zip:
                continue
                
            first_img = seq_imgs_in_zip[0]
            try:
                with z.open(first_img) as img_file:
                    with Image.open(img_file) as img:
                        img_w, img_h = img.size
            except:
                continue
                
            annotations_by_frame = {}
            for line in anno_content.strip().split('\n'):
                parts = line.strip().split(',')
                if len(parts) < 8: continue
                try:
                    frame_idx = int(parts[0])
                    category = int(parts[7])
                except ValueError:
                    continue
                
                if frame_idx % STRIDE_VISDRONE != 1:
                    continue
                    
                if category in [1, 2]: # pedestrian or people
                    left, top, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    cx, cy = (left + w/2)/img_w, (top + h/2)/img_h
                    norm_w, norm_h = w/img_w, h/img_h
                    
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    norm_w = max(0.0, min(1.0, norm_w))
                    norm_h = max(0.0, min(1.0, norm_h))
                    
                    if frame_idx not in annotations_by_frame:
                        annotations_by_frame[frame_idx] = []
                    annotations_by_frame[frame_idx].append(f"0 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")
                    
            for frame_idx, lines in annotations_by_frame.items():
                img_zip_path = f"{prefix}sequences/{seq_name}/{frame_idx:07d}.jpg"
                if img_zip_path not in namelist:
                    fallback_name = f"{frame_idx}.jpg"
                    matched = [f for f in seq_imgs_in_zip if f.endswith(fallback_name)]
                    if matched: img_zip_path = matched[0]
                    else: continue
                
                items.append({
                    "source": split_source,
                    "prefix": f"visdrone_{seq_name}_",
                    "zip_file": zip_path,
                    "img_zip_path": img_zip_path,
                    "label_lines": lines
                })
    dataset_items.extend(items)
    print(f"Added {len(items)} VisDrone items from {zip_path}.")

def process_llvip():
    zip_path = "LLVIP.zip"
    if not Path(zip_path).exists():
        print("LLVIP.zip not found!")
        return
        
    print("Processing LLVIP...")
    items = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = z.namelist()
        
        # Read all XML annotations
        xml_files = [f for f in namelist if f.startswith("LLVIP/Annotations/") and f.endswith(".xml")]
        for xml_file in xml_files:
            stem = Path(xml_file).stem # e.g. "010001"
            
            try:
                xml_content = z.read(xml_file)
                root = ET.fromstring(xml_content)
                size = root.find('size')
                img_w = float(size.find('width').text)
                img_h = float(size.find('height').text)
                
                label_lines = []
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name.lower() == 'person':
                        bndbox = obj.find('bndbox')
                        xmin = float(bndbox.find('xmin').text)
                        ymin = float(bndbox.find('ymin').text)
                        xmax = float(bndbox.find('xmax').text)
                        ymax = float(bndbox.find('ymax').text)
                        
                        w = xmax - xmin
                        h = ymax - ymin
                        cx = xmin + w/2.0
                        cy = ymin + h/2.0
                        
                        # YOLO format
                        nx, ny, nw, nh = cx/img_w, cy/img_h, w/img_w, h/img_h
                        nx = max(0.0, min(1.0, nx))
                        ny = max(0.0, min(1.0, ny))
                        nw = max(0.0, min(1.0, nw))
                        nh = max(0.0, min(1.0, nh))
                        
                        label_lines.append(f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
                
                if not label_lines:
                    continue
                    
                # LLVIP: Only use 'visible' images, skip 'infrared' as requested
                for cam_type in ['visible']:
                    # We don't know if it's in train or test folder, so we search
                    possible_paths = [
                        f"LLVIP/{cam_type}/train/{stem}.jpg",
                        f"LLVIP/{cam_type}/test/{stem}.jpg"
                    ]
                    
                    found_path = None
                    for p in possible_paths:
                        if p in namelist:
                            found_path = p
                            break
                            
                    if found_path:
                        llvip_source = "train" if "/train/" in found_path else "test"
                        items.append({
                            "source": llvip_source,
                            "prefix": f"llvip_{cam_type}_",
                            "zip_file": zip_path,
                            "img_zip_path": found_path,
                            "label_lines": label_lines
                        })
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
                
    dataset_items.extend(items)
    print(f"Added {len(items)} LLVIP items.")

def process_widerperson():
    zip_path = "WiderPerson.zip"
    if not Path(zip_path).exists():
        print("WiderPerson.zip not found!")
        return
        
    print("Processing WiderPerson...")
    items = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = z.namelist()
        
        # Annotations are .jpg.txt
        txt_files = [f for f in namelist if f.startswith("Annotations/") and f.endswith(".txt")]
        for txt_file in txt_files:
            stem = Path(txt_file).name.replace(".jpg.txt", "") # e.g. "000040"
            
            img_path = f"Images/{stem}.jpg"
            if img_path not in namelist:
                continue
                
            try:
                # get dimensions
                with z.open(img_path) as img_file:
                    with Image.open(img_file) as img:
                        img_w, img_h = img.size
                        
                content = z.read(txt_file).decode('utf-8').strip().split('\n')
                if len(content) <= 1: continue
                
                label_lines = []
                # First line is count, skip it
                for line in content[1:]:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if cls_id in [1, 2, 3]: # pedestrian, rider, partial person
                            xmin, ymin, xmax, ymax = map(float, parts[1:5])
                            w = xmax - xmin
                            h = ymax - ymin
                            cx = xmin + w/2.0
                            cy = ymin + h/2.0
                            
                            nx, ny, nw, nh = cx/img_w, cy/img_h, w/img_w, h/img_h
                            nx = max(0.0, min(1.0, nx))
                            ny = max(0.0, min(1.0, ny))
                            nw = max(0.0, min(1.0, nw))
                            nh = max(0.0, min(1.0, nh))
                            
                            label_lines.append(f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
                            
                if label_lines:
                    items.append({
                        "source": "random",
                        "prefix": "wider_",
                        "zip_file": zip_path,
                        "img_zip_path": img_path,
                        "label_lines": label_lines
                    })
            except Exception as e:
                pass
                
    dataset_items.extend(items)
    print(f"Added {len(items)} WiderPerson items.")

def save_splits():
    print(f"Total valid items collected: {len(dataset_items)}")
    
    # Shuffle for random distribution
    random.seed(42)
    random_items = [i for i in dataset_items if i.get("source") == "random"]
    random.shuffle(random_items)
    
    total_rand = len(random_items)
    r_train_end = int(total_rand * 0.8)
    r_val_end = int(total_rand * 0.9)
    
    train_items = [i for i in dataset_items if i.get("source") == "train"] + random_items[:r_train_end]
    val_items = [i for i in dataset_items if i.get("source") == "val"] + random_items[r_train_end:r_val_end]
    test_items = [i for i in dataset_items if i.get("source") == "test"] + random_items[r_val_end:]
    
    splits_map = {
        'train': train_items,
        'val': val_items,
        'test': test_items
    }
    
    print(f"Splitting: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")
    
    # Pre-open zip files to avoid opening/closing repeatedly
    open_zips = {}
    if Path("VisDrone2019-VID-train.zip").exists():
        open_zips["VisDrone2019-VID-train.zip"] = zipfile.ZipFile("VisDrone2019-VID-train.zip", 'r')
    if Path("VisDrone2019-VID-val.zip").exists():
        open_zips["VisDrone2019-VID-val.zip"] = zipfile.ZipFile("VisDrone2019-VID-val.zip", 'r')
    if Path("VisDrone2019-VID-test-dev.zip").exists():
        open_zips["VisDrone2019-VID-test-dev.zip"] = zipfile.ZipFile("VisDrone2019-VID-test-dev.zip", 'r')
    if Path("LLVIP.zip").exists():
        open_zips["LLVIP.zip"] = zipfile.ZipFile("LLVIP.zip", 'r')
    if Path("WiderPerson.zip").exists():
        open_zips["WiderPerson.zip"] = zipfile.ZipFile("WiderPerson.zip", 'r')
        
    for split_name, items in splits_map.items():
        print(f"Writing {split_name} data...")
        img_out_dir = OUTPUT_DIR / 'images' / split_name
        lbl_out_dir = OUTPUT_DIR / 'labels' / split_name
        
        for idx, item in enumerate(items):
            if idx > 0 and idx % 1000 == 0:
                print(f"  {idx}/{len(items)} processed...")
                
            prefix = item.get("prefix", "")
            label_lines = item.get("label_lines", [])
            
            # Determine source and output names
            if "zip_file" in item:
                # Need to extract from zip
                zip_path = item["zip_file"]
                img_zip_path = item["img_zip_path"]
                base_name = f"{prefix}{Path(img_zip_path).name}"
                out_img_path = img_out_dir / base_name
                out_lbl_path = lbl_out_dir / f"{Path(base_name).stem}.txt"
                
                try:
                    img_data = open_zips[zip_path].read(img_zip_path)
                    with open(out_img_path, 'wb') as f:
                        f.write(img_data)
                except Exception as e:
                    print(f"Failed extracting {img_zip_path}: {e}")
                    continue
            else:
                # Local disk file (ExDark)
                img_path = item["img_path"]
                base_name = f"{prefix}{img_path.name}"
                out_img_path = img_out_dir / base_name
                out_lbl_path = lbl_out_dir / f"{Path(base_name).stem}.txt"
                shutil.copy(img_path, out_img_path)
                
            # Write labels
            with open(out_lbl_path, 'w') as f:
                f.write("\n".join(label_lines) + "\n")
                
    for z in open_zips.values():
        z.close()
        
def create_yaml():
    yaml_content = f"""path: /kaggle/input/mega-dataset-final/Mega_Dataset_Final
train: images/train
val: images/val
test: images/test

nc: 1
names: ['person']
"""
    with open('mega.yaml', 'w') as f:
        f.write(yaml_content)
    print("Created mega.yaml")

def zip_mega_dataset():
    out_zip = Path("Mega_Dataset_Final.zip")
    print(f"Zipping {OUTPUT_DIR} to {out_zip} using forward slashes...")
    if out_zip.exists():
        out_zip.unlink()
        
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(OUTPUT_DIR.parent).as_posix()
                zipf.write(file_path, arcname)
    print("Mega_Dataset.zip created successfully!")

if __name__ == '__main__':
    setup_directories()
    process_exdark()
    process_visdrone("VisDrone2019-VID-train.zip")
    process_visdrone("VisDrone2019-VID-val.zip")
    if Path("VisDrone2019-VID-test-dev.zip").exists():
        process_visdrone("VisDrone2019-VID-test-dev.zip")
    process_llvip()
    process_widerperson()
    save_splits()
    create_yaml()
    zip_mega_dataset()
    print("SUCCESS: Mega Dataset is ready!")
