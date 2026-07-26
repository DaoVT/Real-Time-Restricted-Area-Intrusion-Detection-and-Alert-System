"""Tạo manifest thu thập dữ liệu từ video VIRAT.

Công cụ quét thư mục video, suy ra camera từ tên file, tìm polygon tương ứng
trong thư mục zones và ghi danh sách công việc cho collect_dataset.py.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
# Ví dụ VIRAT_S_010204_...: 0102 là camera, 04 là đoạn video của camera đó.
VIRAT_CAMERA_PATTERN = re.compile(r"^VIRAT_S_(\d{4})\d{2}(?:_|$)", re.IGNORECASE)
MANIFEST_COLUMNS = [
    "video_path",
    "zone_path",
    "camera_id",
    "output_name",
    "source_url",
    "source_note",
]


@dataclass(frozen=True)
class ViratManifestRow:
    video_path: Path
    zone_path: Path
    camera_id: str
    output_name: str


def build_virat_manifest(
    raw_root: Path,
    zones_root: Path,
    destination: Path,
    videos_per_camera: int | None = None,
) -> list[ViratManifestRow]:
    """Ghép từng video với đúng file vùng cấm rồi ghi thành manifest."""

    raw_root = Path(raw_root).resolve()
    zones_root = Path(zones_root).resolve()
    destination = Path(destination).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"không tìm thấy thư mục video: {raw_root}")
    if not zones_root.is_dir():
        raise FileNotFoundError(f"không tìm thấy thư mục zone: {zones_root}")
    if videos_per_camera is not None and videos_per_camera < 1:
        raise ValueError("videos_per_camera phải lớn hơn hoặc bằng 1")

    # BƯỚC 1: Tìm tất cả video trong các thư mục con.
    video_paths = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        raise ValueError(f"không tìm thấy video trong: {raw_root}")

    # BƯỚC 2: Suy ra camera và tìm file zone có cùng tên camera.
    rows: list[ViratManifestRow] = []
    output_names: set[str] = set()
    missing_zones: dict[str, Path] = {}
    selected_per_camera: dict[str, int] = {}
    for video_path in video_paths:
        camera_id = infer_camera_id(video_path, raw_root)
        if (
            videos_per_camera is not None
            and selected_per_camera.get(camera_id, 0) >= videos_per_camera
        ):
            continue
        zone_path = zones_root / f"{camera_id}.json"
        if not zone_path.is_file():
            missing_zones[camera_id] = zone_path
            continue

        output_name = _safe_output_name(f"{camera_id}__{video_path.stem}")
        if output_name in output_names:
            raise ValueError(f"output_name bị trùng: {output_name}")
        output_names.add(output_name)
        selected_per_camera[camera_id] = selected_per_camera.get(camera_id, 0) + 1
        rows.append(
            ViratManifestRow(
                video_path=video_path,
                zone_path=zone_path,
                camera_id=camera_id,
                output_name=output_name,
            )
        )

    if missing_zones:
        details = ", ".join(f"{camera}: {path}" for camera, path in sorted(missing_zones.items()))
        raise FileNotFoundError(f"thiếu file zone cho camera: {details}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_path": _relative_or_absolute(row.video_path, destination.parent),
                    "zone_path": _relative_or_absolute(row.zone_path, destination.parent),
                    "camera_id": row.camera_id,
                    "output_name": row.output_name,
                    "source_url": "https://viratdata.org/",
                    "source_note": "VIRAT Ground",
                }
            )
    return rows


def infer_camera_id(video_path: Path, raw_root: Path) -> str:
    """Lấy mã camera từ folder con; nếu video ở root thì đọc tiền tố tên file VIRAT."""

    relative_path = Path(video_path).resolve().relative_to(Path(raw_root).resolve())
    match = VIRAT_CAMERA_PATTERN.match(relative_path.stem)
    if match:
        # Tên gốc VIRAT chứa mã scene/camera; ưu tiên mã này dù file nằm trong folder train/validate.
        return f"VIRAT_S_{match.group(1)}".upper()
    if relative_path.parent != Path("."):
        # Mỗi folder cấp đầu đại diện một góc quay; các video trong folder dùng chung zone.
        return _safe_output_name(relative_path.parts[0]).upper()

    raise ValueError(
        f"không suy ra được camera_id từ {video_path.name}; "
        "hãy đặt video trong folder riêng theo camera"
    )


def _safe_output_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise ValueError(f"tên không hợp lệ: {value!r}")
    return cleaned


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Tạo manifest thu thập mức cao từ video VIRAT đã tải về."
    )
    parser.add_argument("--raw-root", required=True, type=Path, help="Folder chứa video theo camera.")
    parser.add_argument("--zones-root", required=True, type=Path, help="Folder chứa <camera_id>.json.")
    parser.add_argument("--output", required=True, type=Path, help="Đường dẫn manifest CSV cần tạo.")
    parser.add_argument(
        "--videos-per-camera",
        type=int,
        help="Giới hạn số video mỗi camera để tạo pilot cân bằng.",
    )
    args = parser.parse_args()

    try:
        rows = build_virat_manifest(
            args.raw_root,
            args.zones_root,
            args.output,
            videos_per_camera=args.videos_per_camera,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Không thể tạo manifest: {exc}")
        return 1

    camera_count = len({row.camera_id for row in rows})
    print(f"Đã tạo {args.output}: {len(rows)} video thuộc {camera_count} camera.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
