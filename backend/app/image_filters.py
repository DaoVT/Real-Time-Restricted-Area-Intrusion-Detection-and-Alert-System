import cv2
import numpy as np

# --- THRESHOLDS & SETTINGS CỐ ĐỊNH ---
VERY_DARK_THRESHOLD = 40
DARK_THRESHOLD = 75
BLUR_THRESHOLD = 80

CLAHE_CLIP = 1.8
UNSHARP_STRENGTH = 0.6
GAMMA_VALUE = 1.8
# -----------------------------------

def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """
    Tăng cường độ tương phản cục bộ (cân bằng sáng).
    Lợi ích: Cực tốt cho camera quay ban đêm, thiếu sáng, bị ngược sáng.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def apply_unsharp_mask(frame: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Làm nét ảnh mờ bằng Unsharp Masking. Strength được tinh chỉnh để tự nhiên nhất.
    """
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    sharpened = cv2.addWeighted(frame, 1.0 + UNSHARP_STRENGTH, blurred, -UNSHARP_STRENGTH, 0)
    return sharpened

def apply_gamma_correction(frame: np.ndarray) -> np.ndarray:
    """
    Làm sáng video cực tối và khử nhiễu nhẹ để tránh nhiễu hạt do khuếch đại sáng.
    """
    inv_gamma = 1.0 / GAMMA_VALUE
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    processed = cv2.LUT(frame, table)
    return cv2.fastNlMeansDenoisingColored(processed, None, 5, 5, 7, 21)

def detect_best_filter(frame: np.ndarray) -> str:
    """
    Phân tích frame để chọn filter. Thu nhỏ ảnh trước khi phân tích để chạy siêu nhanh.
    """
    if frame is None or frame.size == 0:
        return "none"
        
    height, width = frame.shape[:2]
    scale = min(320 / width, 180 / height, 1.0)
    if scale < 1.0:
        small_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    else:
        small_frame = frame

    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_small)
    
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_metric = cv2.Laplacian(gray_full, cv2.CV_64F).var()
    
    if mean_brightness < VERY_DARK_THRESHOLD:
        if blur_metric < BLUR_THRESHOLD:
            return "gamma_unsharp"
        return "gamma_correction"
    elif mean_brightness < DARK_THRESHOLD:
        if blur_metric < BLUR_THRESHOLD:
            return "clahe_unsharp"
        return "clahe"
    elif blur_metric < BLUR_THRESHOLD:
        return "unsharp_mask"
    
    return "none"

def process_frame(frame: np.ndarray, filter_type: str = "none") -> np.ndarray:
    """
    Hàm tổng hợp để dễ dàng chọn hoặc kết hợp thuật toán tiền xử lý ảnh.
    Chỉ giữ lại những thuật toán mạnh mẽ và cần thiết nhất cho YOLO & CCTV.
    """
    if frame is None or frame.size == 0:
        return frame

    if filter_type == "clahe":
        return apply_clahe(frame)
    elif filter_type == "gamma_correction":
        return apply_gamma_correction(frame)
    elif filter_type == "unsharp_mask":
        return apply_unsharp_mask(frame)
    elif filter_type == "clahe_unsharp":
        processed = apply_clahe(frame)
        return apply_unsharp_mask(processed)
    elif filter_type == "gamma_unsharp":
        processed = apply_gamma_correction(frame)
        return apply_unsharp_mask(processed)
        
    return frame
