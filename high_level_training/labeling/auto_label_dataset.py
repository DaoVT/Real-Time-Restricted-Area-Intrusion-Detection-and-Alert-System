"""Tạo nhãn khởi đầu LOW/MEDIUM/HIGH bằng các luật có thể giải thích.

Mỗi thuộc tính hành vi vượt ngưỡng sẽ cộng điểm rủi ro.
Tổng điểm và các dấu hiệu mạnh quyết định nhãn cuối cùng.
Đây là nhãn yếu để khởi tạo dataset; người dùng vẫn cần xem lại các event khó.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from backend.app.risk.dataset_exporter import EVENT_COLUMNS, SNAPSHOT_COLUMNS


@dataclass(frozen=True)
class RiskRule:
    """Một điều kiện cộng điểm khi thuộc tính hành vi vượt qua ngưỡng."""

    feature: str
    medium_threshold: float
    high_threshold: float
    description: str


# ==========================================================
# BẢNG LUẬT CHẤM ĐIỂM
# Mỗi luật gồm tên thuộc tính, ngưỡng vừa, ngưỡng cao và lý do cộng điểm.
# ==========================================================
RISK_RULES = (
    RiskRule("inside_duration_sec", 8.0, 30.0, "ở trong vùng lâu"),
    RiskRule("average_speed_norm", 0.015, 0.030, "tốc độ trung bình cao"),
    RiskRule("max_speed_norm", 0.050, 0.100, "có thời điểm di chuyển nhanh"),
    RiskRule("max_zone_depth_norm", 0.050, 0.100, "đi sâu vào vùng"),
    RiskRule("stopped_duration_sec", 5.0, 20.0, "đứng lại lâu"),
    RiskRule("direction_change_count", 3.0, 8.0, "đổi hướng nhiều"),
    RiskRule("people_in_zone_peak", 3.0, 5.0, "nhiều người cùng ở trong vùng"),
    # Chỉ cần tái xâm nhập một lần đã là dấu hiệu mạnh nên luật này cộng thẳng 2 điểm.
    RiskRule("reentry_count", 1.0, 1.0, "tái xâm nhập"),
)

MEDIUM_SCORE = 3
HIGH_SCORE = 7
WEAK_LABEL_PREFIX = "Nhãn tự động theo luật v1"
BEHAVIORAL_V2_PREFIX = "Nhãn tự động theo chính sách hành vi v2"
POLICIES = {"points-v1", "behavioral-v2"}
RiskScorer = Callable[[dict[str, str]], tuple[str, int, list[str]]]


def score_risk(row: dict[str, str]) -> tuple[str, int, list[str]]:
    """Cộng điểm của một dòng dữ liệu và ghi lại từng lý do được cộng."""

    score = 0
    reasons: list[str] = []
    for rule in RISK_RULES:
        value = _read_number(row, rule.feature)
        if value >= rule.high_threshold:
            score += 2
            reasons.append(f"{rule.description} (mức cao)")
        elif value >= rule.medium_threshold:
            score += 1
            reasons.append(f"{rule.description} (mức vừa)")

    if score >= HIGH_SCORE:
        label = "HIGH"
    elif score >= MEDIUM_SCORE:
        label = "MEDIUM"
    else:
        label = "LOW"
    return label, score, reasons


def score_behavioral_policy_v2(
    row: dict[str, str],
) -> tuple[str, int, list[str]]:
    """Xét các dấu hiệu hành vi mạnh để chọn LOW, MEDIUM hoặc HIGH."""

    strong_signals = {
        "ở trong vùng từ 20 giây": _read_number(row, "inside_duration_sec") >= 20.0,
        "đứng yên từ 15 giây": _read_number(row, "stopped_duration_sec") >= 15.0,
        "có tái xâm nhập": _read_number(row, "reentry_count") >= 1.0,
        "có ít nhất 5 người trong vùng": _read_number(row, "people_in_zone_peak") >= 5.0,
        "tốc độ cực đại cao": _read_number(row, "max_speed_norm") >= 0.100,
        "đi sâu vào vùng": _read_number(row, "max_zone_depth_norm") >= 0.100,
    }
    contextual_signal = any(
        strong_signals[description]
        for description in (
            "ở trong vùng từ 20 giây",
            "đứng yên từ 15 giây",
            "có tái xâm nhập",
            "có ít nhất 5 người trong vùng",
        )
    )
    strong_reasons = [
        description
        for description, matched in strong_signals.items()
        if matched
    ]
    strong_count = len(strong_reasons)
    if strong_count >= 2 and contextual_signal:
        return "HIGH", strong_count, strong_reasons

    moderate_signals = {
        "ở trong vùng từ 5 giây": _read_number(row, "inside_duration_sec") >= 5.0,
        "độ sâu từ 0,05": _read_number(row, "max_zone_depth_norm") >= 0.050,
        "đứng yên từ 3 giây": _read_number(row, "stopped_duration_sec") >= 3.0,
        "tốc độ cực đại từ 0,05": _read_number(row, "max_speed_norm") >= 0.050,
        "có ít nhất 2 người trong vùng": _read_number(row, "people_in_zone_peak") >= 2.0,
        "đổi hướng từ 3 lần": _read_number(row, "direction_change_count") >= 3.0,
    }
    moderate_reasons = [
        description
        for description, matched in moderate_signals.items()
        if matched
    ]
    if moderate_reasons:
        return "MEDIUM", strong_count, strong_reasons + moderate_reasons
    return "LOW", strong_count, []


def auto_label_dataset(
    events_path: Path,
    snapshots_path: Path,
    output_dir: Path,
    policy: str = "points-v1",
) -> dict[str, Any]:
    """Gán nhãn tự động rồi ghi sang dataset mới, không sửa hai CSV nguồn."""

    if policy not in POLICIES:
        raise ValueError(f"policy phải thuộc {sorted(POLICIES)}")
    scorer: RiskScorer = (
        score_behavioral_policy_v2
        if policy == "behavioral-v2"
        else score_risk
    )
    note_prefix = (
        BEHAVIORAL_V2_PREFIX
        if policy == "behavioral-v2"
        else WEAK_LABEL_PREFIX
    )

    # BƯỚC 1: Đọc hai CSV chưa gán nhãn.
    events = _read_rows(events_path, EVENT_COLUMNS)
    snapshots = _read_rows(snapshots_path, SNAPSHOT_COLUMNS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # BƯỚC 2: Gán nhãn cho event và ghi nhớ camera của từng event.
    camera_by_event: dict[str, str] = {}
    for row in events:
        event_id = row["event_id"].strip()
        camera_id = row["camera_id"].strip()
        if not event_id or not camera_id:
            raise ValueError("event_id và camera_id trong file event không được để trống")
        if event_id in camera_by_event:
            raise ValueError(f"event_id bị trùng: {event_id}")
        camera_by_event[event_id] = camera_id

    event_counts = _label_rows(events, scorer, note_prefix)
    repaired_camera_ids = 0
    for row in snapshots:
        event_id = row["event_id"].strip()
        expected_camera_id = camera_by_event.get(event_id)
        if expected_camera_id is None:
            raise ValueError(f"snapshot không có event tương ứng: {event_id}")
        camera_id = row["camera_id"].strip()
        if not camera_id:
            # Dataset cũ thiếu camera_id ở snapshot; event là nguồn chuẩn để khôi phục.
            row["camera_id"] = expected_camera_id
            repaired_camera_ids += 1
        elif camera_id != expected_camera_id:
            raise ValueError(f"camera_id của snapshot không khớp event {event_id}")
    snapshot_counts = _label_rows(snapshots, scorer, note_prefix)

    events_output = output_dir / "high_level_events.csv"
    snapshots_output = output_dir / "high_level_snapshots.csv"
    _write_rows(events_output, EVENT_COLUMNS, events)
    _write_rows(snapshots_output, SNAPSHOT_COLUMNS, snapshots)

    report: dict[str, Any] = {
        "method": policy,
        "events": len(events),
        "snapshots": len(snapshots),
        "event_label_counts": dict(sorted(event_counts.items())),
        "snapshot_label_counts": dict(sorted(snapshot_counts.items())),
        "repaired_snapshot_camera_ids": repaired_camera_ids,
        "medium_score": MEDIUM_SCORE if policy == "points-v1" else None,
        "high_score": HIGH_SCORE if policy == "points-v1" else None,
        "rules": (
            [asdict(rule) for rule in RISK_RULES]
            if policy == "points-v1"
            else {
                "high": (
                    "Có ít nhất hai dấu hiệu mạnh và có ít nhất một dấu hiệu "
                    "bối cảnh: ở lâu, đứng lâu, tái xâm nhập hoặc đông người."
                ),
                "medium": "Có ít nhất một dấu hiệu mức vừa.",
                "low": "Chưa có dấu hiệu mức vừa hoặc mức cao.",
            }
        ),
        "events_output": str(events_output),
        "snapshots_output": str(snapshots_output),
    }
    report_path = output_dir / "weak_label_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def _label_rows(
    rows: list[dict[str, str]],
    scorer: RiskScorer,
    note_prefix: str,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        label, score, reasons = scorer(row)
        row["label"] = label
        reason_text = "; ".join(reasons) if reasons else "chưa có dấu hiệu làm tăng rủi ro"
        row["annotator_note"] = (
            f"{note_prefix}; số dấu hiệu mạnh={score}; {reason_text}"
            if note_prefix == BEHAVIORAL_V2_PREFIX
            else f"{note_prefix}; điểm={score}; {reason_text}"
        )
        counts[label] += 1
    return counts


def _read_number(row: dict[str, str], feature: str) -> float:
    try:
        return float(row[feature])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{feature} không phải giá trị số hợp lệ") from exc


def _read_rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{Path(path).name} thiếu cột: {sorted(missing)}")
        return list(reader)


def _write_rows(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    # Thêm BOM để Excel trên Windows tự nhận đúng UTF-8 và hiển thị tiếng Việt có dấu.
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tạo weak labels LOW/MEDIUM/HIGH từ đặc trưng hành vi."
    )
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--policy",
        choices=sorted(POLICIES),
        default="points-v1",
    )
    args = parser.parse_args()

    try:
        report = auto_label_dataset(
            args.events,
            args.snapshots,
            args.output_dir,
            policy=args.policy,
        )
    except (OSError, ValueError) as exc:
        # Dùng tiền tố ASCII để thông báo vẫn in được khi PowerShell đang dùng cp1252.
        print(f"Auto-label failed: {exc}")
        return 1
    # Escape ký tự Unicode khi in để CLI không lỗi trên PowerShell đang dùng bảng mã cp1252.
    # File báo cáo vẫn được ghi UTF-8 và giữ nguyên tiếng Việt có dấu.
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
