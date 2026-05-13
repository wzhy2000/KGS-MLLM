from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_PATH = PROJECT_ROOT / "result" / "testdataset_result_batch" / "DLKGS" / "test_results.json"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "image" / "DLKGS"

LABEL_ORDER = ["A", "DR", "H", "IM", "N"]
LABEL_CONFIG = {
    "A": {"classes": [0, 1, 2, 3], "color": "blue"},
    "DR": {"classes": [0, 1, 2, 3], "color": "orange"},
    "H": {"classes": [0, 1, 2], "color": "gold"},
    "IM": {"classes": [0, 1, 2, 3], "color": "purple"},
    "N": {"classes": [0, 1, 2], "color": "deepskyblue"},
}


def get_true_label(sub_category: object) -> int:
    subgroup = str(sub_category).upper()
    if subgroup == "NO":
        return 0

    match = re.search(r"\d+", subgroup)
    if match:
        return int(match.group()) + 1

    return -1


def load_results(json_path: Path) -> list[dict]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unexpected JSON structure: {json_path}")


def collect_category_targets_and_scores(results: list[dict], category: str) -> tuple[list[int], np.ndarray]:
    true_labels: list[int] = []
    predicted_scores: list[list[float]] = []

    for item in results:
        if item.get("category") != category:
            continue

        true_label = get_true_label(item.get("sub_category", ""))
        prediction = item.get("predictions", {}).get(category)
        if true_label < 0 or prediction is None:
            continue

        scores = prediction.get("scores")
        if scores is None:
            continue

        true_labels.append(int(true_label))
        predicted_scores.append([float(score) for score in scores])

    return true_labels, np.array(predicted_scores, dtype=float)


def compute_micro_pr(true_labels: list[int], predicted_scores: np.ndarray, classes: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
    true_labels_bin = label_binarize(true_labels, classes=classes)
    precision, recall, _ = precision_recall_curve(true_labels_bin.ravel(), predicted_scores.ravel())
    pr_auc = auc(recall, precision)
    return precision, recall, float(pr_auc)


def compute_micro_roc(true_labels: list[int], predicted_scores: np.ndarray, classes: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
    true_labels_bin = label_binarize(true_labels, classes=classes)
    fpr, tpr, _ = roc_curve(true_labels_bin.ravel(), predicted_scores.ravel())
    roc_auc = roc_auc_score(true_labels_bin, predicted_scores, average="micro")
    return fpr, tpr, float(roc_auc)


def draw_combined_pr(results: list[dict], output_dir: Path) -> Path:
    plt.figure(figsize=(10, 8))
    for category in LABEL_ORDER:
        true_labels, predicted_scores = collect_category_targets_and_scores(results, category)
        precision, recall, pr_auc = compute_micro_pr(true_labels, predicted_scores, LABEL_CONFIG[category]["classes"])
        plt.plot(
            recall,
            precision,
            linewidth=3,
            color=LABEL_CONFIG[category]["color"],
            label=f"{category} (area = {pr_auc:0.2f})",
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Micro-average Precision-Recall Curve for Five Categories")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    output_path = output_dir / "Multiple_data_sets_P_R_from_json.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def draw_single_category_pr(results: list[dict], category: str, output_dir: Path) -> Path:
    true_labels, predicted_scores = collect_category_targets_and_scores(results, category)
    precision, recall, pr_auc = compute_micro_pr(true_labels, predicted_scores, LABEL_CONFIG[category]["classes"])

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        linewidth=3,
        color=LABEL_CONFIG[category]["color"],
        label=f"Micro-average P-R (area = {pr_auc:0.2f})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{category} Micro-average Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    output_path = output_dir / f"{category}_micro_average_P_R.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def draw_combined_roc(results: list[dict], output_dir: Path) -> Path:
    plt.figure(figsize=(10, 8))
    for category in LABEL_ORDER:
        true_labels, predicted_scores = collect_category_targets_and_scores(results, category)
        fpr, tpr, roc_auc = compute_micro_roc(true_labels, predicted_scores, LABEL_CONFIG[category]["classes"])
        plt.plot(
            fpr,
            tpr,
            linewidth=3,
            color=LABEL_CONFIG[category]["color"],
            label=f"{category} (area = {roc_auc:0.2f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-average ROC Curve for Five Categories")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    output_path = output_dir / "Multiple_data_sets_ROC_from_json.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def draw_single_category_roc(results: list[dict], category: str, output_dir: Path) -> Path:
    true_labels, predicted_scores = collect_category_targets_and_scores(results, category)
    fpr, tpr, roc_auc = compute_micro_roc(true_labels, predicted_scores, LABEL_CONFIG[category]["classes"])

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        linewidth=3,
        color=LABEL_CONFIG[category]["color"],
        label=f"Micro-average ROC (area = {roc_auc:0.2f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{category} Micro-average ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    output_path = output_dir / f"{category}_micro_average_ROC.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def draw_pr_curves(results: list[dict], output_root: Path) -> list[Path]:
    output_dir = output_root / "P_R"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [draw_combined_pr(results, output_dir)]
    outputs.extend(draw_single_category_pr(results, category, output_dir) for category in LABEL_ORDER)
    return outputs


def draw_roc_curves(results: list[dict], output_root: Path) -> list[Path]:
    output_dir = output_root / "ROC"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [draw_combined_roc(results, output_dir)]
    outputs.extend(draw_single_category_roc(results, category, output_dir) for category in LABEL_ORDER)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw DLKGs micro-average ROC and precision-recall curves.")
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--curve", choices=["all", "roc", "pr"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.json_path.exists():
        raise FileNotFoundError(f"json file does not exist: {args.json_path}")

    results = load_results(args.json_path)
    outputs: list[Path] = []
    if args.curve in {"all", "pr"}:
        outputs.extend(draw_pr_curves(results, args.output_root))
    if args.curve in {"all", "roc"}:
        outputs.extend(draw_roc_curves(results, args.output_root))

    for output in outputs:
        print(f"saved: {output}")


if __name__ == "__main__":
    main()
