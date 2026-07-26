"""Chạy nhiều video liên tiếp để thu thập dataset mức cao.

Manifest cho biết video, polygon vùng cấm, mã camera và tên output.
Mỗi video được chạy qua YOLO, ByteTrack, FSM và FeatureExtractor.
Kết quả của từng video nằm trong một thư mục riêng để gộp ở bước sau.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CollectionItem:
    video_path: Path
    zone_path: Path
    camera_id: str
    output_name: str


COMPLETION_MARKER = ".collection_complete.json"


def load_manifest(path: Path) -> list[CollectionItem]:
    """Đọc danh sách công việc và đổi mọi đường dẫn thành đường dẫn tuyệt đối."""

    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"video_path", "zone_path", "camera_id", "output_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest thiếu cột: {sorted(missing)}")

        items: list[CollectionItem] = []
        output_names: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            output_name = row["output_name"].strip()
            if not output_name or Path(output_name).name != output_name:
                raise ValueError(f"output_name không hợp lệ tại dòng {line_number}")
            if output_name in output_names:
                raise ValueError(f"output_name bị trùng tại dòng {line_number}: {output_name}")
            output_names.add(output_name)
            camera_id = row["camera_id"].strip()
            if not camera_id:
                raise ValueError(f"camera_id trống tại dòng {line_number}")
            video_path = _resolve_from_manifest(path, row["video_path"])
            zone_path = _resolve_from_manifest(path, row["zone_path"])
            if not video_path.is_file():
                raise FileNotFoundError(f"không tìm thấy video tại dòng {line_number}: {video_path}")
            if not zone_path.is_file():
                raise FileNotFoundError(f"không tìm thấy zone tại dòng {line_number}: {zone_path}")
            items.append(
                CollectionItem(
                    video_path=video_path,
                    zone_path=zone_path,
                    camera_id=camera_id,
                    output_name=output_name,
                )
            )
    return items


def _resolve_from_manifest(manifest_path: Path, value: str) -> Path:
    candidate = Path(value.strip())
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def main() -> int:
    # PowerShell có thể mặc định CP1252, cần UTF-8 để không lỗi khi in thuật ngữ tiếng Việt.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Chạy nhiều video để thu thập dataset hành vi mức cao.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", default="best.pt", type=str)
    parser.add_argument("--confidence", default=0.35, type=float)
    parser.add_argument("--confirm-frames", default=5, type=int)
    parser.add_argument("--process-width", default=640, type=int)
    parser.add_argument("--frame-stride", default=3, type=int)
    parser.add_argument("--filter-type", default="none", type=str)
    parser.add_argument("--save-output-video", action="store_true")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N video đầu để chạy pilot.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ kiểm tra manifest và in kế hoạch, không tải model hoặc xử lý video.",
    )
    args = parser.parse_args()

    items = load_manifest(args.manifest)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit phải lớn hơn hoặc bằng 1")
        items = items[: args.limit]
    print(f"Đã kiểm tra manifest: {len(items)} video.")
    if args.dry_run:
        for index, item in enumerate(items, start=1):
            print(
                f"[{index}/{len(items)}] {item.camera_id}: "
                f"{item.video_path.name} -> {item.output_name}"
            )
        return 0

    # Chỉ tải YOLO khi xử lý thật; chế độ dry-run chỉ kiểm tra đường dẫn và cấu hình.
    from backend.app.video_processor import PersonDetector, VideoProcessor, load_zone_file

    args.output_root.mkdir(parents=True, exist_ok=True)
    failed = 0
    for index, item in enumerate(items, start=1):
        output_dir = args.output_root / item.output_name
        completion_marker = output_dir / COMPLETION_MARKER
        if completion_marker.is_file():
            print(f"[{index}/{len(items)}] Bỏ qua {item.output_name}: output đã có dữ liệu")
            continue
        if output_dir.exists() and any(output_dir.iterdir()):
            failed += 1
            print(
                f"[{index}/{len(items)}] Bỏ qua {item.output_name}: "
                "output thiếu marker hoàn tất, hãy kiểm tra hoặc chọn output-root mới"
            )
            continue
        print(f"[{index}/{len(items)}] Đang xử lý {item.video_path.name}")
        try:
            # Mỗi video dùng tracker mới để ID của video trước không đi sang video sau.
            detector = PersonDetector(confidence=args.confidence, model_name=args.model)
            result = VideoProcessor(
                input_path=item.video_path,
                output_dir=output_dir,
                zone_points=load_zone_file(item.zone_path),
                confidence=args.confidence,
                confirm_frames=args.confirm_frames,
                detector=detector,
                process_width=args.process_width,
                frame_stride=args.frame_stride,
                save_output_video=args.save_output_video,
                filter_type=args.filter_type,
                camera_id=item.camera_id,
            ).run()
            # Marker xác nhận video đã xử lý xong; thiếu marker nghĩa là lần chạy bị gián đoạn.
            completion_marker.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[{index}/{len(items)}] Hoàn tất: {result['intrusion_count']} sự kiện")
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(items)}] Lỗi {item.video_path.name}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
