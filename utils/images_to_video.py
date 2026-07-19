import cv2
import os
import argparse
from pathlib import Path

def images_to_video(image_folder, output_video, fps=30):
    # Lọc ra các file ảnh và sắp xếp theo thứ tự bảng chữ cái (rất quan trọng với dataset MOT)
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(valid_extensions)]
    images.sort()

    if not images:
        print(f"Error: No images found in '{image_folder}'.")
        return

    # Đọc ảnh đầu tiên để lấy kích thước (width, height)
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    
    if frame is None:
        print(f"Error: Cannot read image '{first_image_path}'.")
        return
        
    height, width, layers = frame.shape

    # Khởi tạo VideoWriter với codec 'avc1' (H.264) thay vì 'mp4v'
    # H.264 là chuẩn bắt buộc để video có thể chạy được trên giao diện Web / Trình duyệt
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"Started merging {len(images)} images into video: {output_video} (FPS={fps})...")
    
    for i, image in enumerate(images):
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Warning: Cannot read {img_path}, skipping.")
            continue
            
        video.write(frame)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(images)} images...")

    cv2.destroyAllWindows()
    video.release()
    print(f"Done! Video saved at: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge image sequence into an MP4 video")
    parser.add_argument("--img-dir", required=True, help="Path to image directory")
    parser.add_argument("--out", required=True, help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.img_dir):
        print(f"Error: Directory '{args.img_dir}' does not exist.")
    else:
        images_to_video(args.img_dir, args.out, args.fps)
