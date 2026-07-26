"""Chuẩn bị dữ liệu MOT để dùng chung pipeline xử lý video.

MOT lưu dữ liệu dưới dạng chuỗi ảnh thay vì MP4.
Công cụ đọc seqinfo.ini, kiểm tra đủ frame, ghép ảnh thành video
và tạo manifest chứa video, vùng cấm cùng mã camera.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


MANIFEST_COLUMNS = [
    "video_path",
    "zone_path",
    "camera_id",
    "output_name",
    "source_url",
    "source_note",
]


@dataclass(frozen=True)
class MotSequenceInfo:
    """Thông tin cần thiết để đổi một chuỗi ảnh MOT thành video."""

    name: str
    image_dir: Path
    frame_rate: float
    sequence_length: int
    width: int
    height: int
    image_extension: str


def load_sequence_info(sequence_dir: Path) -> MotSequenceInfo:
    """Đọc kích thước, FPS và số frame từ file seqinfo.ini."""

    sequence_dir = Path(sequence_dir).resolve()
    metadata_path = sequence_dir / "seqinfo.ini"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"không tìm thấy seqinfo.ini: {metadata_path}")

    parser = configparser.ConfigParser()
    parser.read(metadata_path, encoding="utf-8-sig")
    if "Sequence" not in parser:
        raise ValueError(f"seqinfo.ini thiếu mục [Sequence]: {metadata_path}")

    metadata = parser["Sequence"]
    try:
        name = metadata["name"].strip()
        image_dir = sequence_dir / metadata["imDir"].strip()
        frame_rate = float(metadata["frameRate"])
        sequence_length = int(metadata["seqLength"])
        width = int(metadata["imWidth"])
        height = int(metadata["imHeight"])
        image_extension = metadata["imExt"].strip().lower()
    except (KeyError, ValueError) as exc:
        raise ValueError(f"metadata MOT không hợp lệ: {metadata_path}") from exc

    if not name or frame_rate <= 0 or sequence_length <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"metadata MOT chứa giá trị không hợp lệ: {metadata_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"không tìm thấy thư mục ảnh: {image_dir}")
    if not image_extension.startswith("."):
        image_extension = f".{image_extension}"

    return MotSequenceInfo(
        name=name,
        image_dir=image_dir,
        frame_rate=frame_rate,
        sequence_length=sequence_length,
        width=width,
        height=height,
        image_extension=image_extension,
    )


def list_sequence_frames(info: MotSequenceInfo) -> list[Path]:
    """Sắp xếp ảnh theo thứ tự và dừng nếu dataset bị thiếu frame."""

    frames = sorted(
        path
        for path in info.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == info.image_extension
    )
    if len(frames) != info.sequence_length:
        raise ValueError(
            f"{info.name} có {len(frames)}/{info.sequence_length} frame; "
            "không chuyển đổi dữ liệu tải dở"
        )
    return frames


def convert_sequence_to_video(
    sequence_dir: Path,
    destination: Path,
    overwrite: bool = False,
) -> MotSequenceInfo:
    """Ghép chuỗi frame MOT thành MP4 để dùng lại VideoProcessor hiện có."""

    import cv2

    info = load_sequence_info(sequence_dir)
    frames = list_sequence_frames(info)
    destination = Path(destination).resolve()
    if destination.exists() and not overwrite:
        capture = cv2.VideoCapture(str(destination))
        try:
            existing_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            existing_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            existing_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (
                capture.isOpened()
                and existing_frames == info.sequence_length
                and existing_width == info.width
                and existing_height == info.height
            ):
                return info
        finally:
            capture.release()
        raise FileExistsError(
            f"video đích đã tồn tại nhưng không hợp lệ: {destination}; "
            "dùng --overwrite sau khi kiểm tra"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.frame_rate,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"không thể tạo video: {destination}")

    try:
        for frame_number, frame_path in enumerate(frames, start=1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"không đọc được frame: {frame_path}")
            if frame.shape[1] != info.width or frame.shape[0] != info.height:
                raise ValueError(
                    f"kích thước frame {frame_number} không khớp seqinfo.ini: {frame_path}"
                )
            writer.write(frame)
    finally:
        writer.release()

    # Mở lại file để phát hiện sớm lỗi codec hoặc file video bị hỏng.
    capture = cv2.VideoCapture(str(destination))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"video vừa tạo không mở lại được: {destination}")
        actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if actual_frames != info.sequence_length:
            raise RuntimeError(
                f"video vừa tạo có {actual_frames}/{info.sequence_length} frame: {destination}"
            )
    finally:
        capture.release()
    return info


def prepare_mot_dataset(
    raw_root: Path,
    zones_root: Path,
    videos_root: Path,
    manifest_path: Path,
    selected_sequences: set[str] | None = None,
    overwrite: bool = False,
) -> list[MotSequenceInfo]:
    """Chuyển các sequence đã chọn và tạo manifest cho pipeline mức cao."""

    raw_root = Path(raw_root).resolve()
    zones_root = Path(zones_root).resolve()
    videos_root = Path(videos_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"không tìm thấy dữ liệu MOT: {raw_root}")
    if not zones_root.is_dir():
        raise FileNotFoundError(f"không tìm thấy thư mục zone: {zones_root}")

    sequence_dirs = sorted(path.parent for path in raw_root.rglob("seqinfo.ini"))
    if selected_sequences:
        sequence_dirs = [
            path for path in sequence_dirs if path.name in selected_sequences
        ]
        found_names = {path.name for path in sequence_dirs}
        missing = selected_sequences - found_names
        if missing:
            raise FileNotFoundError(f"không tìm thấy sequence MOT: {sorted(missing)}")
    if not sequence_dirs:
        raise ValueError(f"không tìm thấy sequence MOT trong: {raw_root}")

    prepared: list[MotSequenceInfo] = []
    rows: list[dict[str, str]] = []
    for sequence_dir in sequence_dirs:
        zone_path = zones_root / f"{sequence_dir.name}.json"
        if not zone_path.is_file():
            raise FileNotFoundError(f"thiếu zone cho {sequence_dir.name}: {zone_path}")

        video_path = videos_root / f"{sequence_dir.name}.mp4"
        info = convert_sequence_to_video(sequence_dir, video_path, overwrite=overwrite)
        prepared.append(info)
        rows.append(
            {
                "video_path": str(video_path),
                "zone_path": str(zone_path),
                # Mỗi sequence MOT là một góc quay độc lập trong bước chia dữ liệu.
                "camera_id": info.name.replace("-", "_"),
                "output_name": f"MOT20__{info.name}",
                "source_url": "https://motchallenge.net/data/MOT20Det/",
                "source_note": "MOTChallenge MOT20",
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return prepared


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Chuyển chuỗi ảnh MOT20 thành video và tạo manifest mức cao."
    )
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--zones-root", required=True, type=Path)
    parser.add_argument("--videos-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Manifest CSV cần tạo.")
    parser.add_argument(
        "--sequence",
        action="append",
        dest="sequences",
        help="Tên sequence cần xử lý; có thể truyền nhiều lần.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        prepared = prepare_mot_dataset(
            raw_root=args.raw_root,
            zones_root=args.zones_root,
            videos_root=args.videos_root,
            manifest_path=args.output,
            selected_sequences=set(args.sequences) if args.sequences else None,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Không thể chuẩn bị MOT20: {exc}")
        return 1

    print(f"Đã tạo {args.output} cho {len(prepared)} sequence MOT20.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
