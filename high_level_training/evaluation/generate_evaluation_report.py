"""Tạo các biểu đồ đánh giá cho model Random Forest mức cao.

Công cụ dùng đúng tập test đã chia theo camera, chạy dự đoán rồi tạo
ma trận nhầm lẫn, độ quan trọng thuộc tính, Precision–Recall, ROC,
đường ngưỡng confidence và bảng dự đoán chi tiết.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from high_level_training.modeling.train_random_forest import (
    RISK_LABELS,
    load_labeled_snapshots,
    split_rows_by_assignments,
)


# ==========================================================
# CẤU HÌNH THƯ VIỆN VẼ BIỂU ĐỒ
# Agg cho phép lưu ảnh PNG mà không cần mở cửa sổ đồ họa.
# ==========================================================
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


# Tên tiếng Việt được dùng trên biểu đồ để dễ trình bày và giải thích.
FEATURE_DISPLAY_NAMES = {
    "inside_duration_sec": "Thời gian trong vùng",
    "average_speed_norm": "Tốc độ trung bình",
    "max_speed_norm": "Tốc độ lớn nhất",
    "max_zone_depth_norm": "Độ sâu trong vùng",
    "stopped_duration_sec": "Thời gian đứng yên",
    "reentry_count": "Số lần quay lại",
    "direction_change_count": "Số lần đổi hướng",
    "people_in_zone_peak": "Số người cao nhất",
}
CLASS_COLORS = {
    "LOW": "#2f855a",
    "MEDIUM": "#d69e2e",
    "HIGH": "#c53030",
}


def generate_evaluation_report(
    snapshots_path: Path,
    model_path: Path,
    assignments_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Chạy model trên tập test rồi tạo toàn bộ bảng và biểu đồ đánh giá."""

    # ==========================================================
    # BƯỚC 1: ĐỌC DỮ LIỆU VÀ LẤY ĐÚNG TẬP TEST
    # Dùng lại split_assignments.csv để kết quả không bị thay đổi
    # giữa các lần chạy và không trộn camera của tập train vào test.
    # ==========================================================
    rows, skipped_unlabeled = load_labeled_snapshots(snapshots_path)
    splits = split_rows_by_assignments(rows, assignments_path)
    test_rows = splits["test"]

    # ==========================================================
    # BƯỚC 2: MỞ MODEL ĐÃ TRAIN
    # File joblib chứa Random Forest và danh sách thuộc tính đầu vào.
    # Script chỉ đọc model, không train lại và không ghi đè file model.
    # ==========================================================
    artifact = _load_model_artifact(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    # Chuyển từng dòng CSV thành một mảng gồm 8 thuộc tính số.
    features = np.asarray(
        [[float(row[column]) for column in feature_columns] for row in test_rows],
        dtype=float,
    )

    # ==========================================================
    # BƯỚC 3: CHẠY DỰ ĐOÁN TRÊN TẬP TEST
    # actual: nhãn đúng đã có trong CSV.
    # predicted: nhãn LOW/MEDIUM/HIGH do model chọn.
    # probability_by_label: xác suất model gán cho từng nhãn.
    # ==========================================================
    actual = np.asarray([row["label"] for row in test_rows])
    predicted = np.asarray(model.predict(features))
    raw_probabilities = np.asarray(model.predict_proba(features), dtype=float)
    probability_by_label = _reorder_probabilities(raw_probabilities, model.classes_)
    matrix = confusion_matrix(actual, predicted, labels=list(RISK_LABELS))

    # ==========================================================
    # BƯỚC 4: TẠO CÁC FILE BÁO CÁO
    # Mỗi hàm bên dưới tạo một bảng CSV hoặc một ảnh PNG riêng.
    # ==========================================================
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "test_predictions.csv"
    _write_test_predictions(
        prediction_path,
        test_rows,
        actual,
        predicted,
        probability_by_label,
    )
    _plot_confusion_matrix(output_dir / "confusion_matrix.png", matrix)
    _plot_feature_importance(
        output_dir / "feature_importance.png",
        feature_columns,
        np.asarray(model.feature_importances_, dtype=float),
    )
    average_precision = _plot_precision_recall(
        output_dir / "precision_recall_curve.png",
        actual,
        probability_by_label,
    )
    roc_auc = _plot_roc(
        output_dir / "roc_curve.png",
        actual,
        probability_by_label,
    )
    best_thresholds = _plot_confidence_thresholds(
        output_dir / "confidence_threshold_curve.png",
        actual,
        probability_by_label,
    )
    _plot_class_distribution(
        output_dir / "class_distribution.png",
        splits,
    )

    # ==========================================================
    # BƯỚC 5: LƯU SỐ LIỆU TỔNG HỢP
    # JSON giữ các con số gốc để có thể kiểm tra lại biểu đồ.
    # ==========================================================
    report = {
        "model_path": str(Path(model_path)),
        "snapshots_path": str(Path(snapshots_path)),
        "assignments_path": str(Path(assignments_path)),
        "evaluated_split": "test",
        "evaluated_snapshots": len(test_rows),
        "skipped_unlabeled_rows": skipped_unlabeled,
        "feature_columns": feature_columns,
        "class_counts": dict(Counter(actual)),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=list(RISK_LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "best_f1_threshold": best_thresholds,
        "outputs": {
            "test_predictions": str(prediction_path),
            "confusion_matrix": str(output_dir / "confusion_matrix.png"),
            "feature_importance": str(output_dir / "feature_importance.png"),
            "precision_recall_curve": str(output_dir / "precision_recall_curve.png"),
            "roc_curve": str(output_dir / "roc_curve.png"),
            "confidence_threshold_curve": str(output_dir / "confidence_threshold_curve.png"),
            "class_distribution": str(output_dir / "class_distribution.png"),
        },
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _load_model_artifact(model_path: Path) -> dict[str, Any]:
    """Mở gói model và kiểm tra các thành phần cần cho việc đánh giá."""

    # Một số phiên bản NumPy phát cảnh báo khi đọc cây quyết định cũ.
    # Cảnh báo này không làm thay đổi trọng số hoặc kết quả dự đoán.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("artifact model không phải dictionary")
    model = artifact.get("model")
    feature_columns = artifact.get("feature_columns")
    if model is None or not isinstance(feature_columns, list):
        raise ValueError("artifact model thiếu model hoặc feature_columns")
    if not hasattr(model, "predict_proba") or not hasattr(model, "feature_importances_"):
        raise ValueError("model không hỗ trợ xác suất hoặc độ quan trọng thuộc tính")
    if {str(label) for label in model.classes_} != set(RISK_LABELS):
        raise ValueError("model không có đủ ba nhãn LOW/MEDIUM/HIGH")
    return artifact


def _reorder_probabilities(probabilities: np.ndarray, model_classes: Any) -> np.ndarray:
    """Sắp xếp ba cột xác suất theo thứ tự LOW, MEDIUM, HIGH."""

    # Scikit-learn thường sắp nhãn theo bảng chữ cái: HIGH, LOW, MEDIUM.
    # Đổi lại thứ tự giúp mọi biểu đồ và file CSV dùng cùng một quy ước.
    class_indexes = {str(label): index for index, label in enumerate(model_classes)}
    return np.column_stack(
        [probabilities[:, class_indexes[label]] for label in RISK_LABELS]
    )


def _write_test_predictions(
    path: Path,
    rows: list[dict[str, str]],
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Lưu kết quả của từng snapshot để có thể tìm và xem lại dự đoán sai."""

    fieldnames = [
        "event_id",
        "video_id",
        "camera_id",
        "track_id",
        "video_time_sec",
        "actual_label",
        "predicted_label",
        "prediction_confidence",
        "probability_low",
        "probability_medium",
        "probability_high",
        "is_correct",
    ]
    # UTF-8 BOM giúp Excel trên Windows đọc đúng tiếng Việt.
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "event_id": row["event_id"],
                    "video_id": row["video_id"],
                    "camera_id": row["camera_id"],
                    "track_id": row["track_id"],
                    "video_time_sec": row.get("video_time_sec", ""),
                    "actual_label": actual[index],
                    "predicted_label": predicted[index],
                    "prediction_confidence": float(np.max(probabilities[index])),
                    "probability_low": probabilities[index, 0],
                    "probability_medium": probabilities[index, 1],
                    "probability_high": probabilities[index, 2],
                    "is_correct": actual[index] == predicted[index],
                }
            )


def _plot_confusion_matrix(path: Path, matrix: np.ndarray) -> None:
    """Vẽ bảng đếm nhãn thật theo hàng và nhãn model dự đoán theo cột."""

    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title="Ma trận nhầm lẫn trên tập test",
        xlabel="Nhãn model dự đoán",
        ylabel="Nhãn thực tế",
        xticks=range(len(RISK_LABELS)),
        yticks=range(len(RISK_LABELS)),
        xticklabels=RISK_LABELS,
        yticklabels=RISK_LABELS,
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "#102a43",
                fontsize=11,
                fontweight="bold",
            )
    _save_figure(figure, path)


def _plot_feature_importance(
    path: Path,
    feature_columns: list[str],
    importances: np.ndarray,
) -> None:
    """Vẽ mức độ Random Forest sử dụng từng thuộc tính khi ra quyết định."""

    ordered = sorted(
        zip(feature_columns, importances),
        key=lambda item: item[1],
    )
    names = [FEATURE_DISPLAY_NAMES.get(name, name) for name, _ in ordered]
    values = [float(value) for _, value in ordered]
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    bars = axis.barh(names, values, color="#2878b5")
    axis.set(
        title="Độ quan trọng của thuộc tính",
        xlabel="Mức đóng góp tương đối",
    )
    axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    axis.set_xlim(0, max(values, default=1.0) * 1.18)
    _save_figure(figure, path)


def _plot_precision_recall(
    path: Path,
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Đo sự cân bằng giữa phát hiện đủ sự kiện và hạn chế cảnh báo sai."""

    from sklearn.metrics import precision_recall_curve

    figure, axis = plt.subplots(figsize=(8.0, 6.2))
    scores: dict[str, float] = {}
    for class_index, label in enumerate(RISK_LABELS):
        # Đánh giá lần lượt từng nhãn: nhãn đang xét là đúng, hai nhãn còn lại là không đúng.
        binary_actual = (actual == label).astype(int)
        precision, recall, _ = precision_recall_curve(
            binary_actual,
            probabilities[:, class_index],
        )
        score = float(
            average_precision_score(binary_actual, probabilities[:, class_index])
        )
        scores[label] = score
        axis.plot(
            recall,
            precision,
            color=CLASS_COLORS[label],
            linewidth=2,
            label=f"{label} (AP={score:.3f})",
        )
    axis.set(
        title="Precision–Recall theo từng mức rủi ro",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, path)
    return scores


def _plot_roc(
    path: Path,
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Đo khả năng tách một mức rủi ro khỏi hai mức còn lại."""

    figure, axis = plt.subplots(figsize=(8.0, 6.2))
    scores: dict[str, float] = {}
    for class_index, label in enumerate(RISK_LABELS):
        # Ví dụ khi xét HIGH: HIGH là nhóm cần phát hiện, LOW và MEDIUM là nhóm còn lại.
        binary_actual = (actual == label).astype(int)
        false_positive_rate, true_positive_rate, _ = roc_curve(
            binary_actual,
            probabilities[:, class_index],
        )
        score = float(
            roc_auc_score(binary_actual, probabilities[:, class_index])
        )
        scores[label] = score
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            color=CLASS_COLORS[label],
            linewidth=2,
            label=f"{label} (AUC={score:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#718096", label="Ngẫu nhiên")
    axis.set(
        title="ROC–AUC theo từng mức rủi ro",
        xlabel="Tỷ lệ báo động giả",
        ylabel="Tỷ lệ phát hiện đúng",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, path)
    return scores


def _plot_confidence_thresholds(
    path: Path,
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Thử các ngưỡng từ 0 đến 1 để tìm ngưỡng có F1 tốt nhất cho từng nhãn."""

    # Mỗi bước tăng 0,01: 0,00; 0,01; ...; 1,00.
    thresholds = np.linspace(0.0, 1.0, 101)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    best_by_label: dict[str, dict[str, float]] = {}
    for class_index, (axis, label) in enumerate(zip(axes, RISK_LABELS)):
        binary_actual = actual == label
        precision_values: list[float] = []
        recall_values: list[float] = []
        f1_values: list[float] = []
        for threshold in thresholds:
            # Chỉ chấp nhận nhãn khi xác suất của nó lớn hơn hoặc bằng ngưỡng đang thử.
            binary_prediction = probabilities[:, class_index] >= threshold
            true_positive = int(np.sum(binary_prediction & binary_actual))
            false_positive = int(np.sum(binary_prediction & ~binary_actual))
            false_negative = int(np.sum(~binary_prediction & binary_actual))
            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 1.0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

        # Chọn vị trí có F1 cao nhất để đánh dấu bằng đường nét đứt trên biểu đồ.
        best_index = int(np.argmax(f1_values))
        best_by_label[label] = {
            "threshold": float(thresholds[best_index]),
            "f1": float(f1_values[best_index]),
            "precision": float(precision_values[best_index]),
            "recall": float(recall_values[best_index]),
        }
        axis.plot(thresholds, precision_values, label="Precision", color="#2878b5")
        axis.plot(thresholds, recall_values, label="Recall", color="#2f855a")
        axis.plot(thresholds, f1_values, label="F1", color="#c53030", linewidth=2)
        axis.axvline(
            thresholds[best_index],
            color="#4a5568",
            linestyle="--",
            alpha=0.8,
        )
        axis.set_title(f"{label}: F1 tốt nhất tại {thresholds[best_index]:.2f}")
        axis.set_xlabel("Ngưỡng confidence")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Điểm số")
    axes[-1].legend(loc="lower left")
    figure.suptitle("Precision, Recall và F1 theo ngưỡng confidence")
    _save_figure(figure, path)
    return best_by_label


def _plot_class_distribution(
    path: Path,
    splits: dict[str, list[dict[str, str]]],
) -> None:
    """Đếm số snapshot LOW/MEDIUM/HIGH trong train, validation và test."""

    split_names = ("train", "validation", "test")
    positions = np.arange(len(split_names))
    bar_width = 0.24
    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    for class_index, label in enumerate(RISK_LABELS):
        values = [
            Counter(row["label"] for row in splits[split_name])[label]
            for split_name in split_names
        ]
        offset = (class_index - 1) * bar_width
        bars = axis.bar(
            positions + offset,
            values,
            width=bar_width,
            color=CLASS_COLORS[label],
            label=label,
        )
        axis.bar_label(bars, padding=3, fontsize=8)
    axis.set(
        title="Phân bố nhãn theo tập dữ liệu",
        xlabel="Tập dữ liệu",
        ylabel="Số snapshot",
        xticks=positions,
        xticklabels=("Train", "Validation", "Test"),
    )
    axis.legend()
    _save_figure(figure, path)


def _save_figure(figure: Any, path: Path) -> None:
    """Căn lại bố cục, lưu ảnh PNG rồi giải phóng bộ nhớ của biểu đồ."""

    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Tạo biểu đồ đánh giá cho model phân loại rủi ro mức cao.",
    )
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = generate_evaluation_report(
        snapshots_path=args.snapshots,
        model_path=args.model,
        assignments_path=args.assignments,
        output_dir=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
