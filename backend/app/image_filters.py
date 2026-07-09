import cv2
import numpy as np

def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Tăng cường độ tương phản cục bộ (cân bằng sáng).
    Lợi ích: Cực tốt cho camera quay ban đêm, thiếu sáng, bị ngược sáng. Giúp hiện rõ bóng người trong bóng tối.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def apply_histogram_equalization(frame: np.ndarray) -> np.ndarray:
    """
    Cân bằng sáng toàn cục.
    Lợi ích: Tăng tương phản mạnh. Nhược điểm: Dễ bị cháy sáng ở những vùng đã sáng sẵn (ưu tiên dùng CLAHE hơn).
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))
    channels[0] = cv2.equalizeHist(channels[0])
    return cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCrCb2BGR)

def apply_gaussian_blur(frame: np.ndarray, kernel_size: tuple = (5, 5)) -> np.ndarray:
    """
    Khử nhiễu hột (noise).
    Lợi ích: Rất tốt cho camera thiếu sáng bị nhiễu hột hoặc ảnh ISO cao. Giảm thiểu sai số nhận diện (False Positives).
    """
    return cv2.GaussianBlur(frame, kernel_size, 0)

def apply_median_filter(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Lọc nhiễu muối tiêu (muỗi).
    Lợi ích: Khử các điểm nhiễu lốm đốm rất tốt nhưng vẫn giữ lại được viền (edge) sắc nét của người.
    """
    return cv2.medianBlur(frame, kernel_size)

def apply_sharpening(frame: np.ndarray) -> np.ndarray:
    """
    Tăng độ nét cho ảnh bị mờ.
    Lợi ích: Phục hồi chi tiết cho các camera độ phân giải thấp hoặc ống kính mờ.
    """
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1], 
                       [-1, -1, -1]])
    return cv2.filter2D(frame, -1, kernel)

def apply_unsharp_mask(frame: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    """
    Mặt nạ làm nét (Unsharp Masking) - Thuật toán làm nét hiệu quả nhất cho video.
    Nguyên lý: Lấy ảnh gốc trừ đi ảnh đã làm mờ (để tìm viền), sau đó cộng viền đó vào ảnh gốc.
    Lợi ích: Làm video "nét căng" mà ít bị nhiễu hạt hơn so với Laplacian (apply_sharpening).
    """
    # 1. Làm mờ ảnh để lấy phần nền (không có viền sắc)
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    # 2. Cộng dồn: original * (1 + strength) - blurred * strength
    sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
    return sharpened

def apply_gamma_correction(frame: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """
    Gamma Correction: Làm sáng các vùng tối mà không làm "cháy" lóa các vùng đã sáng sẵn.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

def process_frame(frame: np.ndarray, filter_type: str = "none") -> np.ndarray:
    """
    Hàm tổng hợp để dễ dàng chọn hoặc kết hợp nhiều thuật toán.
    """
    if filter_type == "clahe":
        return apply_clahe(frame)
    elif filter_type == "hist_eq":
        return apply_histogram_equalization(frame)
    elif filter_type == "gaussian":
        return apply_gaussian_blur(frame)
    elif filter_type == "median":
        return apply_median_filter(frame)
    elif filter_type == "sharpen":
        return apply_sharpening(frame)
    elif filter_type == "unsharp_mask":
        return apply_unsharp_mask(frame, sigma=1.0, strength=1.5)
    elif filter_type == "clahe_gaussian": 
        # Kết hợp: Đầu tiên giảm nhiễu hạt, sau đó kéo sáng để vùng tối hiện rõ
        return apply_clahe(apply_gaussian_blur(frame, (3, 3)))
    elif filter_type == "clahe_median":
        # Kết hợp tối ưu nhất cho nhiễu muối tiêu (salt and pepper): Median Blur + CLAHE
        return apply_clahe(apply_median_filter(frame, kernel_size=5))
    elif filter_type == "restore_night_camera":
        # 1. Khử nhiễu hột cực mạnh mà không làm mờ viền (Bilateral Filter + Median)
        # (GaussianBlur làm mờ tịt (blur) cả người, ta đổi sang Bilateral để giữ viền sắc nét)
        smoothed = cv2.medianBlur(frame, 3) # Khử bớt các hạt nhiễu gai góc
        smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 2. Kéo sáng lên (nhân 2.5 lần để bù lại mức 40% bị giảm lúc tạo video)
        brightened = cv2.convertScaleAbs(smoothed, alpha=2.5, beta=10)
        
        # Lưu ý: BỎ qua CLAHE ở đây vì CLAHE có nhược điểm chí mạng là "phóng đại nhiễu" ở các vùng tối.
        # Dùng CLAHE sau khi nhân sáng sẽ khiến các hạt nhiễu li ti bùng lên.
        return brightened
    elif filter_type == "super_sharp":
        # Bước 1: Khử nhiễu (Denoising)
        # Tăng nhẹ tham số để lọc nhiễu tốt hơn mà vẫn giữ viền
        denoised = cv2.bilateralFilter(frame, d=5, sigmaColor=35, sigmaSpace=35)

        # Bước 2: Tăng cường độ sáng và độ tương phản (Low-light Enhancement)
        # Chuyển sang không gian màu LAB để tách biệt độ sáng (L) và màu sắc (A, B)
        lab_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image)

        # Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) lên kênh L
        # Giúp kéo sáng các vùng tối mà không làm lóa các đèn đường
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Gộp lại và chuyển về BGR
        merged_lab = cv2.merge((cl, a, b))
        brightened = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        # Bước 3: Tăng độ sắc nét (Unsharp Masking)
        # Lực làm nét cân bằng để tránh nhiễu và vệt sáng giả (sử dụng 2.4 và -1.4)
        gaussian_blur = cv2.GaussianBlur(brightened, (0, 0), 1.5)
        sharpened = cv2.addWeighted(brightened, 2.4, gaussian_blur, -1.4, 0)

        # Bước 4: Hiệu chỉnh Gamma để làm sáng toàn bộ video hơn một chút (áp dụng cho tất cả video)
        final_frame = apply_gamma_correction(sharpened, gamma=1.3)

        return final_frame
    
    # "none"
    return frame
