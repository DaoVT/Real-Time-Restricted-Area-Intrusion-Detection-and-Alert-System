import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# Tự động chuyển thư mục làm việc về thư mục gốc dự án
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
sys.path.append(str(project_root))

def main():
    parser = argparse.ArgumentParser(description="Đánh giá chi tiết các chỉ số chất lượng (Metrics) của mô hình YOLO")
    parser.add_argument("--model", type=str, default="test_and_model/kaggle/input/notebooks/minhnguyncng19072005/notebook560d537205/runs/mega_model/weights/best.pt", help="Đường dẫn đến file weights của mô hình (.pt)")
    parser.add_argument("--data", type=str, default="test_and_model/kaggle/input/notebooks/minhnguyncng19072005/notebook560d537205/mega_dataset/data.yaml", help="Đường dẫn file cấu hình dữ liệu (.yaml)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Tập dữ liệu dùng để chạy đánh giá")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào để suy luận")
    
    args = parser.parse_args()
    
    # Kiểm tra sự tồn tại của mô hình
    if not os.path.exists(args.model):
        print(f"❌ ERROR: Không tìm thấy tệp trọng số mô hình tại: {args.model}")
        return
        
    # Kiểm tra sự tồn tại của file cấu hình data
    if not os.path.exists(args.data):
        # Thử tìm file data trong các thư mục con khác
        alt_data = os.path.join("mega_dataset", "data.yaml")
        if os.path.exists(alt_data):
            args.data = alt_data
        else:
            print(f"❌ ERROR: Không tìm thấy tệp cấu hình dữ liệu tại: {args.data}")
            return
            
    print(f"=== ĐANG CHẠY ĐÁNH GIÁ CHỈ SỐ MÔ HÌNH ===")
    print(f"📁 Tệp weights:     {args.model}")
    print(f"📄 Cấu hình data:   {args.data}")
    print(f"📊 Tập đánh giá:    {args.split.upper()}")
    print(f"🖼️ Kích thước ảnh:  {args.imgsz}x{args.imgsz}")
    print("="*50)
    
    # Nạp mô hình YOLO
    model = YOLO(args.model)
    
    # Chạy validation/evaluation
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        verbose=False,
        workers=0
    )
    
    # Trích xuất và in các chỉ số
    print("\n" + "="*50)
    print("                KẾT QUẢ ĐÁNH GIÁ MTRICS")
    print("="*50)
    
    map50 = metrics.box.map50
    map95 = metrics.box.map  # mAP@50-95
    
    # Precision & Recall (lấy cho class person ở index 0)
    p = metrics.box.p[0] if len(metrics.box.p) > 0 else 0.0
    r = metrics.box.r[0] if len(metrics.box.r) > 0 else 0.0
    
    # Tính F1-Score
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    print(f"🔹 mAP@50:       {map50:.4f}  (Khả năng phát hiện ở IoU = 0.5)")
    print(f"🔹 mAP@50-95:    {map95:.4f}  (Chỉ số mAP trung bình tiêu chuẩn COCO)")
    print(f"🔹 Precision:    {p:.4f}  (Độ chính xác của các dự đoán)")
    print(f"🔹 Recall:       {r:.4f}  (Khả năng phát hiện đầy đủ đối tượng)")
    print(f"🔹 F1-Score:     {f1:.4f}  (Điểm cân bằng điều hòa F1)")
    
    # Tốc độ xử lý
    speed = metrics.speed
    preprocess_speed = speed.get('preprocess', 0.0)
    inference_speed = speed.get('inference', 0.0)
    postprocess_speed = speed.get('postprocess', 0.0)
    total_speed = preprocess_speed + inference_speed + postprocess_speed
    
    print("\n" + "="*50)
    print("                THÔNG SỐ TỐC ĐỘ XỬ LÝ (MỖI KHUNG HÌNH)")
    print("="*50)
    print(f"⚡ Tiền xử lý (Preprocess):  {preprocess_speed:.2f} ms")
    print(f"⚡ Nhận diện (Inference):     {inference_speed:.2f} ms")
    print(f"⚡ Hậu xử lý (Postprocess):   {postprocess_speed:.2f} ms")
    print(f"⚡ Tổng thời gian xử lý:     {total_speed:.2f} ms")
    print(f"⚡ FPS ước tính (Inference): {1000/total_speed:.1f} FPS")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
