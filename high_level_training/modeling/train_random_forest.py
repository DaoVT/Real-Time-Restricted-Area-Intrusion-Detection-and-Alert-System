"""Train model Random Forest để phân loại LOW/MEDIUM/HIGH.

Luồng xử lý:
1. Đọc các snapshot đã có nhãn.
2. Chia train, validation và test theo camera để tránh rò rỉ dữ liệu.
3. Train 300 cây quyết định trên 8 thuộc tính hành vi.
4. Đánh giá model và lưu model, chỉ số cùng phép chia dữ liệu.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


RISK_LABELS = ("LOW", "MEDIUM", "HIGH")

# ==========================================================
# 8 THUỘC TÍNH ĐƯỢC ĐƯA VÀO MODEL
# Không dùng confidence YOLO và số lần quan sát vì hai giá trị này thay đổi
# theo chất lượng detector, FPS và chế độ bỏ frame.
# ==========================================================
MODEL_FEATURES = (
    "inside_duration_sec",
    "average_speed_norm",
    "max_speed_norm",
    "max_zone_depth_norm",
    "stopped_duration_sec",
    "reentry_count",
    "direction_change_count",
    "people_in_zone_peak",
)


def load_labeled_snapshots(path: Path) -> tuple[list[dict[str, str]], int]:
    """Đọc snapshot hợp lệ và đếm số dòng chưa có nhãn bị bỏ qua."""

    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required_columns = {"event_id", "video_id", "camera_id", "label", *MODEL_FEATURES}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"dataset thiếu cột: {sorted(missing_columns)}")

        rows: list[dict[str, str]] = []
        skipped_unlabeled = 0
        for line_number, row in enumerate(reader, start=2):
            label = row["label"].strip().upper()
            if not label:
                skipped_unlabeled += 1
                continue
            if label not in RISK_LABELS:
                raise ValueError(f"nhãn không hợp lệ tại dòng {line_number}: {row['label']}")
            row["label"] = label
            for feature in MODEL_FEATURES:
                try:
                    value = float(row[feature])
                except ValueError as exc:
                    raise ValueError(f"{feature} không phải số tại dòng {line_number}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{feature} không hữu hạn tại dòng {line_number}")
            rows.append(row)
    return rows, skipped_unlabeled


def split_rows_by_source(
    rows: list[dict[str, str]],
    random_state: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """Chia nguyên camera vào train, validation hoặc test."""

    # Không chia ngẫu nhiên từng dòng vì các snapshot cùng camera rất giống nhau.
    # Một camera chỉ được phép xuất hiện trong đúng một tập dữ liệu.
    if not rows:
        raise ValueError("dataset không có snapshot đã gán nhãn")
    rows_by_group: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_group.setdefault(_source_group(row), []).append(row)
    if len(rows_by_group) < 3:
        raise ValueError("cần ít nhất 3 camera/video độc lập để tạo train, validation và test")

    labels_by_group = {
        group: {row["label"] for row in group_rows}
        for group, group_rows in rows_by_group.items()
    }
    for label in RISK_LABELS:
        source_count = sum(label in labels for labels in labels_by_group.values())
        if source_count < 3:
            raise ValueError(
                f"nhãn {label} chỉ xuất hiện ở {source_count} camera/video; "
                "không thể chia đủ nhãn cho ba tập độc lập"
            )

    group_names = sorted(rows_by_group)
    random.Random(random_state).shuffle(group_names)
    total_rows = len(rows)
    global_counts = Counter(row["label"] for row in rows)
    group_sizes = {group: len(group_rows) for group, group_rows in rows_by_group.items()}
    group_label_counts = {
        group: Counter(row["label"] for row in group_rows)
        for group, group_rows in rows_by_group.items()
    }

    # Chọn trước các ứng viên test tốt nhất để không phải duyệt toàn bộ 3^N cách chia.
    test_candidates = _candidate_group_subsets(
        group_names,
        rows_by_group,
        target_rows=total_rows * 0.15,
    )[:128]
    best: tuple[float, set[str], set[str]] | None = None
    all_groups = set(group_names)
    minimum_train_groups = max(3, math.ceil(len(group_names) * 0.40))
    for _, test_groups in test_candidates:
        if len(test_groups) < 2:
            continue
        remaining = [group for group in group_names if group not in test_groups]
        for _, validation_groups in _candidate_group_subsets(
            remaining,
            rows_by_group,
            target_rows=total_rows * 0.15,
        )[:128]:
            train_groups = all_groups - test_groups - validation_groups
            if (
                len(validation_groups) < 2
                or len(train_groups) < minimum_train_groups
                or not _contains_all_labels(train_groups, labels_by_group)
            ):
                continue
            score = _split_score(
                train_groups,
                validation_groups,
                test_groups,
                group_sizes,
                group_label_counts,
                global_counts,
                total_rows,
            )
            if best is None or score < best[0]:
                best = (score, validation_groups, test_groups)

    if best is None:
        raise ValueError("không tìm được cách chia theo camera mà cả ba tập đều có đủ nhãn")

    _, validation_groups, test_groups = best
    split_by_group = {
        group: (
            "test"
            if group in test_groups
            else "validation"
            if group in validation_groups
            else "train"
        )
        for group in group_names
    }
    splits = {"train": [], "validation": [], "test": []}
    for row in rows:
        splits[split_by_group[_source_group(row)]].append(row)
    _validate_event_isolation(splits)
    return splits


def _candidate_group_subsets(
    group_names: list[str],
    rows_by_group: dict[str, list[dict[str, str]]],
    target_rows: float,
) -> list[tuple[float, set[str]]]:
    """Liệt kê nhóm camera có đủ nhãn, xếp gần kích thước mục tiêu lên trước."""

    candidates: list[tuple[float, set[str]]] = []
    for subset_size in range(1, len(group_names)):
        for subset_tuple in combinations(group_names, subset_size):
            subset = set(subset_tuple)
            labels = {
                row["label"]
                for group in subset
                for row in rows_by_group[group]
            }
            if not set(RISK_LABELS).issubset(labels):
                continue
            row_count = sum(len(rows_by_group[group]) for group in subset)
            candidates.append((abs(row_count - target_rows), subset))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _contains_all_labels(
    groups: set[str],
    labels_by_group: dict[str, set[str]],
) -> bool:
    labels = {label for group in groups for label in labels_by_group[group]}
    return set(RISK_LABELS).issubset(labels)


def _split_score(
    train_groups: set[str],
    validation_groups: set[str],
    test_groups: set[str],
    group_sizes: dict[str, int],
    group_label_counts: dict[str, Counter[str]],
    global_counts: Counter[str],
    total_rows: int,
) -> float:
    """Ưu tiên tỷ lệ 70/15/15 và phân bố nhãn gần với toàn dataset."""

    targets = {"train": 0.70, "validation": 0.15, "test": 0.15}
    groups_by_split = {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups,
    }
    score = 0.0
    for split_name, groups in groups_by_split.items():
        split_size = sum(group_sizes[group] for group in groups)
        score += abs(split_size / total_rows - targets[split_name])
        split_counts: Counter[str] = Counter()
        for group in groups:
            split_counts.update(group_label_counts[group])
        for label in RISK_LABELS:
            global_ratio = global_counts[label] / total_rows
            split_ratio = split_counts[label] / split_size
            score += 0.20 * abs(split_ratio - global_ratio)
    return score


def train_random_forest(
    snapshots_path: Path,
    output_dir: Path,
    random_state: int = 42,
    assignments_path: Path | None = None,
) -> dict[str, Any]:
    """Train Random Forest và lưu mọi file cần để kiểm tra lại kết quả."""

    # BƯỚC 1: Đọc dữ liệu đã gán nhãn và chia thành ba tập độc lập.
    rows, skipped_unlabeled = load_labeled_snapshots(snapshots_path)
    splits = (
        split_rows_by_assignments(rows, assignments_path)
        if assignments_path is not None
        else split_rows_by_source(rows, random_state=random_state)
    )
    for split_name, split_rows in splits.items():
        if not split_rows:
            raise ValueError(f"tập {split_name} đang rỗng")

    # BƯỚC 2: Đổi các dòng CSV thành X (8 thuộc tính) và y (nhãn đúng).
    train_x, train_y = _to_xy(splits["train"])

    # BƯỚC 3: Train 300 cây. balanced tăng trọng số cho nhãn có ít mẫu.
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)

    # BƯỚC 4: Lưu model cùng tên thuộc tính để backend biết đúng thứ tự đầu vào.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "risk_model.joblib"
    artifact = {
        "model": model,
        "feature_columns": list(MODEL_FEATURES),
        "risk_labels": list(RISK_LABELS),
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "random_state": random_state,
    }
    joblib.dump(artifact, model_path)

    # BƯỚC 5: Đánh giá riêng trên validation và test rồi lưu báo cáo.
    metrics = {
        "model_path": str(model_path),
        "feature_columns": list(MODEL_FEATURES),
        "skipped_unlabeled_rows": skipped_unlabeled,
        "split_strategy": (
            f"fixed:{assignments_path}"
            if assignments_path is not None
            else "grouped-balanced-search"
        ),
        "split_sizes": {name: len(split_rows) for name, split_rows in splits.items()},
        "split_camera_counts": {
            name: len({_source_group(row) for row in split_rows})
            for name, split_rows in splits.items()
        },
        "split_label_counts": {
            name: dict(Counter(row["label"] for row in split_rows))
            for name, split_rows in splits.items()
        },
        "validation": _evaluate(model, splits["validation"]),
        "test": _evaluate(model, splits["test"]),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_confusion_matrix(output_dir / "confusion_matrix.csv", metrics["test"]["confusion_matrix"])
    _write_feature_importance(output_dir / "feature_importance.csv", model.feature_importances_)
    _write_split_assignments(output_dir / "split_assignments.csv", splits)
    return metrics


def split_rows_by_assignments(
    rows: list[dict[str, str]],
    assignments_path: Path,
) -> dict[str, list[dict[str, str]]]:
    """Dùng lại phép chia cũ để hai model được đánh giá trên cùng dữ liệu."""

    with Path(assignments_path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"event_id", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"split assignments thiếu cột: {sorted(missing)}")
        split_by_event = {
            row["event_id"].strip(): row["split"].strip()
            for row in reader
        }

    allowed_splits = {"train", "validation", "test"}
    splits: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    source_owner: dict[str, str] = {}
    for row in rows:
        event_id = row["event_id"].strip()
        split_name = split_by_event.get(event_id)
        if split_name not in allowed_splits:
            raise ValueError(f"event {event_id} không có split hợp lệ")
        source = _source_group(row)
        owner = source_owner.setdefault(source, split_name)
        if owner != split_name:
            raise ValueError(f"camera/video {source} xuất hiện ở cả {owner} và {split_name}")
        splits[split_name].append(row)

    for split_name, split_rows in splits.items():
        labels = {row["label"] for row in split_rows}
        if not set(RISK_LABELS).issubset(labels):
            raise ValueError(f"tập {split_name} không có đủ LOW/MEDIUM/HIGH")
    _validate_event_isolation(splits)
    return splits


def _source_group(row: dict[str, str]) -> str:
    camera_id = row["camera_id"].strip()
    return f"camera:{camera_id}" if camera_id else f"video:{row['video_id'].strip()}"


def _validate_event_isolation(splits: dict[str, list[dict[str, str]]]) -> None:
    event_owners: dict[str, str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            event_id = row["event_id"]
            owner = event_owners.setdefault(event_id, split_name)
            if owner != split_name:
                raise RuntimeError(f"event_id {event_id} bị chia vào cả {owner} và {split_name}")


def _to_xy(rows: list[dict[str, str]]) -> tuple[list[list[float]], list[str]]:
    features = [[float(row[column]) for column in MODEL_FEATURES] for row in rows]
    labels = [row["label"] for row in rows]
    return features, labels


def _evaluate(model: RandomForestClassifier, rows: list[dict[str, str]]) -> dict[str, Any]:
    features, labels = _to_xy(rows)
    predictions = model.predict(features)
    report = classification_report(
        labels,
        predictions,
        labels=list(RISK_LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=list(RISK_LABELS)).tolist()
    return {"classification_report": report, "confusion_matrix": matrix}


def _write_confusion_matrix(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *RISK_LABELS])
        for label, values in zip(RISK_LABELS, matrix):
            writer.writerow([label, *values])


def _write_feature_importance(path: Path, importances: Any) -> None:
    pairs = sorted(zip(MODEL_FEATURES, importances), key=lambda item: item[1], reverse=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["feature", "importance"])
        writer.writerows(pairs)


def _write_split_assignments(path: Path, splits: dict[str, list[dict[str, str]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["event_id", "video_id", "camera_id", "split"])
        writer.writeheader()
        seen_events: set[str] = set()
        for split_name, rows in splits.items():
            for row in rows:
                if row["event_id"] in seen_events:
                    continue
                seen_events.add(row["event_id"])
                writer.writerow(
                    {
                        "event_id": row["event_id"],
                        "video_id": row["video_id"],
                        "camera_id": row["camera_id"],
                        "split": split_name,
                    }
                )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train Random Forest đánh giá rủi ro xâm nhập.")
    parser.add_argument("--snapshots", required=True, type=Path, help="File high_level_snapshots.csv đã gán nhãn.")
    parser.add_argument("--output", required=True, type=Path, help="Thư mục lưu model và báo cáo.")
    parser.add_argument("--random-state", default=42, type=int, help="Seed để tái lập cách chia dữ liệu và model.")
    parser.add_argument(
        "--assignments",
        type=Path,
        help="Tùy chọn: dùng lại split_assignments.csv của model trước.",
    )
    args = parser.parse_args()

    metrics = train_random_forest(
        args.snapshots,
        args.output,
        args.random_state,
        assignments_path=args.assignments,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
