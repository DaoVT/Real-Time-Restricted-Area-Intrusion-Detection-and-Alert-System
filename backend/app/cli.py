from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .video_processor import PersonDetector, VideoProcessor, load_zone_file


def main() -> int:
    # Ép UTF-8 để PowerShell Windows in được phần trợ giúp tiếng Việt có dấu.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run restricted area intrusion detection on a video.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--zone", required=True, type=Path)
    parser.add_argument("--output", default=Path("backend/output/cli"), type=Path)
    parser.add_argument("--confidence", default=0.35, type=float)
    parser.add_argument("--confirm-frames", default=5, type=int)
    parser.add_argument("--process-width", default=None, type=int)
    parser.add_argument("--frame-stride", default=1, type=int)
    parser.add_argument("--no-output-video", action="store_true")
    parser.add_argument("--filter-type", default="auto", type=str, help="Image preprocessing filter type (auto, clahe, gaussian, etc.)")
    # Cho phép kiểm thử bằng weights cụ thể mà không thay đổi model mặc định của backend/UI.
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="Đường dẫn model YOLO. Mặc định ưu tiên best.pt rồi đến yolov8n.pt.",
    )
    parser.add_argument("--camera-id", default="", type=str, help="Mã góc quay do người thu thập dataset tự đặt.")
    args = parser.parse_args()

    try:
        result = VideoProcessor(
            input_path=args.input,
            output_dir=args.output,
            zone_points=load_zone_file(args.zone),
            confidence=args.confidence,
            confirm_frames=args.confirm_frames,
            process_width=args.process_width,
            frame_stride=args.frame_stride,
            save_output_video=not args.no_output_video,
            filter_type=args.filter_type,
            detector=PersonDetector(confidence=args.confidence, model_name=args.model),
            camera_id=args.camera_id,
        ).run()
    except Exception as exc:
        print(f"Video processing failed: {exc}")
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
