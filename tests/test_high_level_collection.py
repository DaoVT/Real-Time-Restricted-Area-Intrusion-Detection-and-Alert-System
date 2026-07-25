from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from high_level_training.data.collect_dataset import COMPLETION_MARKER, load_manifest
from high_level_training.data.prepare_mot import (
    convert_sequence_to_video,
    list_sequence_frames,
    load_sequence_info,
    prepare_mot_dataset,
)
from high_level_training.data.prepare_virat import build_virat_manifest, infer_camera_id


def test_load_manifest_resolves_paths_and_metadata(tmp_path):
    video_path = tmp_path / "video.mp4"
    zone_path = tmp_path / "zone.json"
    video_path.write_bytes(b"video")
    zone_path.write_text("[]", encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "video_path,zone_path,camera_id,output_name\n"
        "video.mp4,zone.json,WEB_CAM_001,run_001\n",
        encoding="utf-8",
    )

    [item] = load_manifest(manifest_path)

    assert item.video_path == video_path
    assert item.zone_path == zone_path
    assert item.camera_id == "WEB_CAM_001"
    assert item.output_name == "run_001"


def test_load_manifest_rejects_duplicate_output_name(tmp_path):
    for filename in ("one.mp4", "two.mp4"):
        (tmp_path / filename).write_bytes(b"video")
    (tmp_path / "zone.json").write_text("[]", encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "video_path,zone_path,camera_id,output_name\n"
        "one.mp4,zone.json,CAM_001,same_output\n"
        "two.mp4,zone.json,CAM_001,same_output\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_name bị trùng"):
        load_manifest(manifest_path)


def test_prepare_virat_builds_manifest_from_camera_folders(tmp_path):
    raw_root = tmp_path / "raw"
    zones_root = tmp_path / "zones"
    camera_dir = raw_root / "VIRAT_S_000001"
    camera_dir.mkdir(parents=True)
    zones_root.mkdir()
    video_path = camera_dir / "clip_001.mp4"
    video_path.write_bytes(b"video")
    zone_path = zones_root / "VIRAT_S_000001.json"
    zone_path.write_text("[]", encoding="utf-8")
    manifest_path = tmp_path / "manifests" / "virat.csv"

    rows = build_virat_manifest(raw_root, zones_root, manifest_path)

    assert len(rows) == 1
    assert rows[0].camera_id == "VIRAT_S_000001"
    with manifest_path.open(encoding="utf-8") as file:
        [manifest_row] = list(csv.DictReader(file))
    assert manifest_row["camera_id"] == "VIRAT_S_000001"
    assert manifest_row["source_note"] == "VIRAT Ground"


def test_prepare_virat_reports_missing_zone(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "VIRAT_S_000002_clip.mp4").write_bytes(b"video")
    zones_root = tmp_path / "zones"
    zones_root.mkdir()

    with pytest.raises(FileNotFoundError, match="VIRAT_S_0000"):
        build_virat_manifest(raw_root, zones_root, tmp_path / "manifest.csv")


def test_prepare_virat_limits_video_count_per_camera(tmp_path):
    raw_root = tmp_path / "raw"
    zones_root = tmp_path / "zones"
    raw_root.mkdir()
    zones_root.mkdir()
    for index in range(3):
        (raw_root / f"VIRAT_S_01020{index}_00_000000_000100.mp4").write_bytes(b"video")
    (zones_root / "VIRAT_S_0102.json").write_text("[]", encoding="utf-8")

    rows = build_virat_manifest(
        raw_root,
        zones_root,
        tmp_path / "manifest.csv",
        videos_per_camera=2,
    )

    assert len(rows) == 2
    assert {row.camera_id for row in rows} == {"VIRAT_S_0102"}


def test_infer_camera_id_uses_virat_filename_at_root(tmp_path):
    video_path = tmp_path / "VIRAT_S_010204_00_000060_000120.mp4"
    assert infer_camera_id(video_path, tmp_path) == "VIRAT_S_0102"


def test_infer_camera_id_prefers_filename_inside_dataset_split(tmp_path):
    video_path = tmp_path / "train" / "VIRAT_S_010205_00_000060_000120.mp4"
    assert infer_camera_id(video_path, tmp_path) == "VIRAT_S_0102"


def test_collection_completion_marker_has_stable_filename():
    assert COMPLETION_MARKER == ".collection_complete.json"


def _create_mot_sequence(tmp_path, frame_count: int = 3):
    sequence_dir = tmp_path / "raw" / "train" / "MOT20-01"
    image_dir = sequence_dir / "img1"
    image_dir.mkdir(parents=True)
    (sequence_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        "name=MOT20-01\n"
        "imDir=img1\n"
        "frameRate=25\n"
        f"seqLength={frame_count}\n"
        "imWidth=32\n"
        "imHeight=24\n"
        "imExt=.jpg\n",
        encoding="utf-8",
    )
    for index in range(1, frame_count + 1):
        frame = np.full((24, 32, 3), index * 20, dtype=np.uint8)
        assert cv2.imwrite(str(image_dir / f"{index:06d}.jpg"), frame)
    return sequence_dir


def test_prepare_mot_reads_metadata_and_converts_video(tmp_path):
    sequence_dir = _create_mot_sequence(tmp_path)
    destination = tmp_path / "videos" / "MOT20-01.mp4"

    info = convert_sequence_to_video(sequence_dir, destination)

    assert info.name == "MOT20-01"
    assert info.sequence_length == 3
    capture = cv2.VideoCapture(str(destination))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        capture.release()


def test_prepare_mot_rejects_incomplete_sequence(tmp_path):
    sequence_dir = _create_mot_sequence(tmp_path, frame_count=3)
    (sequence_dir / "img1" / "000003.jpg").unlink()

    with pytest.raises(ValueError, match="2/3 frame"):
        list_sequence_frames(load_sequence_info(sequence_dir))


def test_prepare_mot_creates_collection_manifest(tmp_path):
    sequence_dir = _create_mot_sequence(tmp_path)
    zones_root = tmp_path / "zones"
    zones_root.mkdir()
    (zones_root / "MOT20-01.json").write_text("[]", encoding="utf-8")
    manifest_path = tmp_path / "manifests" / "pilot.csv"

    prepared = prepare_mot_dataset(
        raw_root=sequence_dir.parents[1],
        zones_root=zones_root,
        videos_root=tmp_path / "videos",
        manifest_path=manifest_path,
        selected_sequences={"MOT20-01"},
    )

    assert [info.name for info in prepared] == ["MOT20-01"]
    with manifest_path.open(encoding="utf-8") as file:
        [row] = list(csv.DictReader(file))
    assert row["camera_id"] == "MOT20_01"
    assert row["output_name"] == "MOT20__MOT20-01"
    assert row["source_note"] == "MOTChallenge MOT20"
