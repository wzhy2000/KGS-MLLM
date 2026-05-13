from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "testdataset"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "table"

CATEGORIES = ["A", "DR", "H", "IM", "N"]
MODEL_DIRS = {
    "result_claude": "result_claude",
    "result_gemini": "result_gemini",
    "result_gpt": "result_gpt",
    "result_grok": "result_grok",
}
DOCTOR_GROUPS = {
    "group_1_doctor1_5": [f"doctor{i}" for i in range(1, 6)],
    "group_2_doctor6_10": [f"doctor{i}" for i in range(6, 11)],
}

CATEGORY_CONFIG = {
    "A": {
        "labels": [0, 1, 2, 3],
        "all_name": "A-all",
        "map_fn": lambda label: "0"
        if label in ["A0", "C-0", "C-1"]
        else (
            "1"
            if label in ["A1", "C-2", "C-3"]
            else ("2" if label in ["A2", "O-1", "O-2", "O-3"] else ("3" if label == "NO" else None))
        ),
        "truth_map_fn": lambda value: "3" if value == "NO" else value,
    },
    "DR": {
        "labels": [0, 1, 2, 3],
        "all_name": "DR-all",
        "map_fn": lambda label: "0"
        if label == "DR-0"
        else ("1" if label == "DR-1" else ("2" if label == "DR-2" else ("3" if label == "NO" else None))),
        "truth_map_fn": lambda value: "3" if value == "NO" else value,
    },
    "H": {
        "labels": [0, 1, 2],
        "all_name": "H-all",
        "map_fn": lambda label: "0" if label == "H-0" else ("1" if label == "H-1" else ("2" if label == "NO" else None)),
        "truth_map_fn": lambda value: "2" if value == "NO" else value,
    },
    "IM": {
        "labels": [0, 1, 2, 3],
        "all_name": "IM-all",
        "map_fn": lambda label: "0"
        if label == "IM-0"
        else ("1" if label == "IM-1" else ("2" if label == "IM-2" else ("3" if label == "NO" else None))),
        "truth_map_fn": lambda value: "3" if value == "NO" else value,
    },
    "N": {
        "labels": [0, 1, 2],
        "all_name": "N-all",
        "map_fn": lambda label: "0" if label == "N-0" else ("1" if label == "N-1" else ("2" if label == "NO" else None)),
        "truth_map_fn": lambda value: "2" if str(value).upper() == "NO" else value,
    },
}

DLKGS_SCORE_MAP = {
    "A": {0: "NO", 1: "A0", 2: "A1", 3: "A2"},
    "DR": {0: "NO", 1: "DR-0", 2: "DR-1", 3: "DR-2"},
    "H": {0: "NO", 1: "H-0", 2: "H-1"},
    "IM": {0: "NO", 1: "IM-0", 2: "IM-1", 3: "IM-2"},
    "N": {0: "NO", 1: "N-0", 2: "N-1"},
}


def binary_specificity(y_true_binary: list[int], y_pred_binary: list[int]) -> float:
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn = cm[0, 0]
    fp = cm[0, 1]
    return (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0


def multiclass_macro_specificity(y_true: list[int], y_pred: list[int], labels: list[int]) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total_samples = cm.sum()
    specificity_scores: list[float] = []

    for idx in range(len(labels)):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total_samples - tp - fp - fn
        specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
        specificity_scores.append(specificity)

    return sum(specificity_scores) / len(specificity_scores) if specificity_scores else 0.0


def load_truth_labels(dataset_root: Path, category: str) -> dict[str, str]:
    truth_path = dataset_root / category / "image_scores.json"
    return json.loads(truth_path.read_text(encoding="utf-8"))


def normalize_image(value: object) -> str:
    text = str(value).replace("\\", "/")
    return text.split(".")[0] + ".jpg"


def build_targets_and_predictions(category: str, df: pd.DataFrame, dataset_root: Path) -> tuple[list[int], list[int]]:
    config = CATEGORY_CONFIG[category]
    truth_labels = load_truth_labels(dataset_root, category)

    y_true: list[int] = []
    y_pred: list[int] = []

    for _, row in df.iterrows():
        image = normalize_image(row["image"])
        pred = config["map_fn"](str(row["score"]).strip())
        truth = truth_labels.get(image)
        if truth is None or pred is None:
            continue

        true_mapped = config["truth_map_fn"](truth)
        y_true.append(int(true_mapped))
        y_pred.append(int(pred))

    return y_true, y_pred


def calculate_overall_metrics(category: str, y_true: list[int], y_pred: list[int]) -> dict[str, float | int | str]:
    config = CATEGORY_CONFIG[category]
    return {
        "name": config["all_name"],
        "samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "specificity": multiclass_macro_specificity(y_true, y_pred, config["labels"]),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
    }


def format_metric_row(name: str, samples: int, accuracy: float, specificity: float, precision: float, recall: float, f1: float) -> str:
    return (
        f"{name:<18}{samples:>7}"
        f"{accuracy:>12.2f}%{specificity:>14.2f}%{precision:>12.2f}%"
        f"{recall:>10.2f}%{f1:>10.2f}%"
    )


def build_overall_row(category_rows: dict[str, dict[str, float | int | str]]) -> dict[str, float | int | str]:
    rows = [category_rows[category] for category in CATEGORIES]
    total_samples = sum(int(row["samples"]) for row in rows)
    return {
        "name": "overall",
        "samples": total_samples,
        "accuracy": sum(float(row["accuracy"]) * int(row["samples"]) for row in rows) / total_samples,
        "specificity": sum(float(row["specificity"]) * int(row["samples"]) for row in rows) / total_samples,
        "precision": sum(float(row["precision"]) * int(row["samples"]) for row in rows) / total_samples,
        "recall": sum(float(row["recall"]) * int(row["samples"]) for row in rows) / total_samples,
        "f1": sum(float(row["f1"]) * int(row["samples"]) for row in rows) / total_samples,
    }


def dlkgs_image_value(item: dict) -> str:
    filepath = Path(item["filepath"])
    parts = filepath.parts
    lowered = [part.lower() for part in parts]
    if "testdataset" in lowered:
        idx = lowered.index("testdataset")
        return "\\".join(parts[idx + 1 :])
    return "\\".join([item["category"], item.get("sub_category", ""), item.get("filename", "")])


def load_dlkgs_results(result_root: Path) -> pd.DataFrame:
    json_path = result_root / "testdataset_result_batch" / "DLKGS" / "test_results.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    results = payload["results"] if isinstance(payload, dict) and "results" in payload else payload

    rows: list[dict[str, str]] = []
    for item in results:
        category = item.get("category")
        if category not in CATEGORIES:
            continue

        prediction = item.get("predictions", {}).get(category, {})
        predicted_label = prediction.get("predicted_label")
        if predicted_label is None:
            continue

        rows.append(
            {
                "image": dlkgs_image_value(item),
                "score": DLKGS_SCORE_MAP[category][int(predicted_label)],
            }
        )

    return pd.DataFrame(rows)


def build_model_group_report(result_root: Path, dataset_root: Path) -> dict[str, dict[str, dict[str, float | int | str]]]:
    report: dict[str, dict[str, dict[str, float | int | str]]] = {}
    batch_root = result_root / "testdataset_result_batch"

    for block_name, model_dir_name in MODEL_DIRS.items():
        report[block_name] = {}
        for category in CATEGORIES:
            workbook_path = batch_root / model_dir_name / f"{category}_batch_result" / "result_batch.xlsx"
            df = pd.read_excel(workbook_path)
            y_true, y_pred = build_targets_and_predictions(category, df, dataset_root)
            report[block_name][category] = calculate_overall_metrics(category, y_true, y_pred)

    dlkgs_df = load_dlkgs_results(result_root)
    report["result_DLKGS"] = {}
    for category in CATEGORIES:
        subset = dlkgs_df[dlkgs_df["image"].astype(str).str.startswith(f"{category}\\")].copy()
        y_true, y_pred = build_targets_and_predictions(category, subset, dataset_root)
        report["result_DLKGS"][category] = calculate_overall_metrics(category, y_true, y_pred)

    return report


def build_doctor_group_report(result_root: Path, dataset_root: Path) -> dict[str, dict[str, dict[str, float | int | str]]]:
    doctor_root = result_root / "result_doctor"
    all_overall_rows: dict[str, dict[str, dict[str, float | int | str]]] = defaultdict(dict)

    doctor_dirs = sorted(
        [path for path in doctor_root.iterdir() if path.is_dir() and path.name.startswith("doctor")],
        key=lambda path: int(path.name.replace("doctor", "")),
    )

    for doctor_dir in doctor_dirs:
        for category in CATEGORIES:
            workbook_path = doctor_dir / f"{category}_result.xlsx"
            df = pd.read_excel(workbook_path)
            y_true, y_pred = build_targets_and_predictions(category, df, dataset_root)
            all_overall_rows[doctor_dir.name][category] = calculate_overall_metrics(category, y_true, y_pred)

    group_averages: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for group_name, doctors in DOCTOR_GROUPS.items():
        group_averages[group_name] = {}
        for category in CATEGORIES:
            rows = [all_overall_rows[doctor][category] for doctor in doctors]
            group_averages[group_name][category] = {
                "name": CATEGORY_CONFIG[category]["all_name"],
                "samples": round(sum(int(row["samples"]) for row in rows) / len(rows)),
                "accuracy": sum(float(row["accuracy"]) for row in rows) / len(rows),
                "specificity": sum(float(row["specificity"]) for row in rows) / len(rows),
                "precision": sum(float(row["precision"]) for row in rows) / len(rows),
                "recall": sum(float(row["recall"]) for row in rows) / len(rows),
                "f1": sum(float(row["f1"]) for row in rows) / len(rows),
            }

    group_averages["groups_3"] = {}
    for category in CATEGORIES:
        row_1 = group_averages["group_1_doctor1_5"][category]
        row_2 = group_averages["group_2_doctor6_10"][category]
        group_averages["groups_3"][category] = {
            "name": CATEGORY_CONFIG[category]["all_name"],
            "samples": round((int(row_1["samples"]) + int(row_2["samples"])) / 2),
            "accuracy": (float(row_1["accuracy"]) + float(row_2["accuracy"])) / 2,
            "specificity": (float(row_1["specificity"]) + float(row_2["specificity"])) / 2,
            "precision": (float(row_1["precision"]) + float(row_2["precision"])) / 2,
            "recall": (float(row_1["recall"]) + float(row_2["recall"])) / 2,
            "f1": (float(row_1["f1"]) + float(row_2["f1"])) / 2,
        }

    return group_averages


def write_group_report(path: Path, title: str, rows_by_block: dict[str, dict[str, dict[str, float | int | str]]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * 92 + "\n\n")
        for block_name, category_rows in rows_by_block.items():
            f.write(f"[{block_name}]\n")
            f.write("Sub-category        Samples    Accuracy   Specificity   Precision    Recall        F1\n")
            for category in CATEGORIES:
                f.write(format_metric_row(**category_rows[category]) + "\n")
            f.write(format_metric_row(**build_overall_row(category_rows)) + "\n")
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build overall classification performance tables from project result files."
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_report = build_model_group_report(args.result_root, args.dataset_root)
    doctor_report = build_doctor_group_report(args.result_root, args.dataset_root)

    model_output = args.output_dir / "model_group_average_report.txt"
    doctor_output = args.output_dir / "doctor_group_average_report.txt"

    write_group_report(model_output, "Model Group Average Report (X-all only)", model_report)
    write_group_report(doctor_output, "Doctor Group Average Report (X-all only)", doctor_report)

    print(f"saved: {model_output}")
    print(f"saved: {doctor_output}")


if __name__ == "__main__":
    main()
