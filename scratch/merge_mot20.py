import os
import shutil
import cv2
import configparser
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    print("Starting MOT20 test download...")
    # Tải thư mục test từ dataset
    repo_dir = snapshot_download(
        repo_id="Lekim89/MOT20",
        repo_type="dataset",
        allow_patterns=["test/*"]
    )
    
    test_dir = Path(repo_dir) / "test"
    seqs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    
    output_file = "mot20_test_merged.mp4"
    target_width, target_height = 1920, 1080
    target_fps = 25
    
    # Khởi tạo VideoWriter với codec hỗ trợ web, ưu tiên H264 nếu có DLL
    fourcc = cv2.VideoWriter_fourcc(*"avc1") 
    writer = cv2.VideoWriter(output_file, fourcc, target_fps, (target_width, target_height))
    
    if not writer.isOpened():
        print("Warning: avc1 codec not available, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_file, fourcc, target_fps, (target_width, target_height))

    for seq_dir in seqs:
        seqinfo_path = seq_dir / "seqinfo.ini"
        if not seqinfo_path.exists():
            continue
            
        config = configparser.ConfigParser()
        config.read(seqinfo_path)
        
        try:
            fps = float(config["Sequence"]["frameRate"])
            imDir = config["Sequence"]["imDir"]
        except Exception:
            fps = target_fps
            imDir = "img1"
            
        print(f"Processing {seq_dir.name} (Original FPS: {fps})")
        
        img_dir = seq_dir / imDir
        images = sorted([f for f in img_dir.iterdir() if f.suffix == ".jpg"])
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Resize image to target resolution maintaining aspect ratio (padding)
            h, w = img.shape[:2]
            scale = min(target_width/w, target_height/h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            
            # Create black canvas and paste
            canvas = cv2.copyMakeBorder(
                resized,
                top=(target_height - new_h) // 2,
                bottom=(target_height - new_h) - (target_height - new_h) // 2,
                left=(target_width - new_w) // 2,
                right=(target_width - new_w) - (target_width - new_w) // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
            
            writer.write(canvas)
            
    writer.release()
    print(f"Successfully created {output_file}")
    
    print("Cleaning up downloaded files...")
    cache_dir = Path(repo_dir).parent.parent
    if "huggingface" in str(cache_dir) and "datasets--Lekim89--MOT20" in str(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print("Deleted HuggingFace cache for MOT20.")
        except Exception as e:
            print(f"Cannot delete cache folder: {e}")
    else:
        print(f"Please delete manually: {cache_dir}")

if __name__ == "__main__":
    main()
