from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "testdataset"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "table"

CATEGORIES = ["A", "DR", "H", "IM", "N"]
TEST_MODEL_CONFIGS = {
    "KGS-Gemini 3": "result_gemini",
    "KGS-GPT 5.2": "result_gpt",
    "KGS-Claude 4.5": "result_claude",
    "KGS-Grok 4": "result_grok",
}
EXTERNAL_MODEL_CONFIGS = {
    "KGS-Gemini 3": "gemini",
}
DOCTOR_GROUPS = {
    "Senior": [f"doctor{i}" for i in range(1, 6)],
    "Junior": [f"doctor{i}" for i in range(6, 11)],
}

CATEGORY_CONFIG = {
    "A": {
        "labels": [0, 1, 2, 3],
        "class_names": {0: "A0", 1: "A1", 2: "A2", 3: "A-N"},
        "paper_order": ["A-N", "A0", "A1", "A2"],
        "score_to_label": {
            "A0": 0,
            "A1": 1,
            "A2": 2,
            "C-0": 0,
            "C-1": 0,
            "C-2": 1,
            "C-3": 1,
            "O-1": 2,
            "O-2": 2,
            "O-3": 2,
            "NO": 3,
        },
        "truth_to_label": lambda value: 3 if str(value).upper() == "NO" else int(value),
    },
    "DR": {
        "labels": [0, 1, 2, 3],
        "class_names": {0: "DR0", 1: "DR1", 2: "DR2", 3: "DR-N"},
        "paper_order": ["DR-N", "DR0", "DR1", "DR2"],
        "score_to_label": {"DR-0": 0, "DR-1": 1, "DR-2": 2, "DR0": 0, "DR1": 1, "DR2": 2, "NO": 3},
        "truth_to_label": lambda value: 3 if str(value).upper() == "NO" else int(value),
    },
    "H": {
        "labels": [0, 1, 2],
        "class_names": {0: "H0", 1: "H1", 2: "H-N"},
        "paper_order": ["H-N", "H0", "H1"],
        "score_to_label": {"H-0": 0, "H-1": 1, "H0": 0, "H1": 1, "NO": 2},
        "truth_to_label": lambda value: 2 if str(value).upper() == "NO" else int(value),
    },
    "IM": {
        "labels": [0, 1, 2, 3],
        "class_names": {0: "IM0", 1: "IM1", 2: "IM2", 3: "IM-N"},
        "paper_order": ["IM-N", "IM0", "IM1", "IM2"],
        "score_to_label": {"IM-0": 0, "IM-1": 1, "IM-2": 2, "IM0": 0, "IM1": 1, "IM2": 2, "NO": 3},
        "truth_to_label": lambda value: 3 if str(value).upper() == "NO" else int(value),
    },
    "N": {
        "labels": [0, 1, 2],
        "class_names": {0: "N0", 1: "N1", 2: "N-N"},
        "paper_order": ["N-N", "N0", "N1"],
        "score_to_label": {"N-0": 0, "N-1": 1, "N0": 0, "N1": 1, "NO": 2},
        "truth_to_label": lambda value: 2 if str(value).upper() == "NO" else int(value),
    },
}


def normalize_truth_key(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    parts = [part for part in text.split("/") if part]
    for index, part in enumerate(parts):
        if part in CATEGORIES:
            return "/".join(parts[index:]).rsplit(".", 1)[0] + ".jpg"
    return text.rsplit(".", 1)[0] + ".jpg"


def load_truth_labels(dataset_root: Path, category: str) -> dict[str, object]:
    truth_path = dataset_root / category / "image_scores.json"
    return json.loads(truth_path.read_text(encoding="utf-8"))


def binary_specificity(y_true_binary: list[int], y_pred_binary: list[int]) -> float:
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn = cm[0, 0]
    fp = cm[0, 1]
    return (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0


def subclass_metric_rows(category: str, y_true: list[int], y_pred: list[int]) -> list[dict[str, float | int | str]]:
    config = CATEGORY_CONFIG[category]
    rows: dict[str, dict[str, float | int | str]] = {}
    for cls in config["labels"]:
        y_true_binary = [1 if value == cls else 0 for value in y_true]
        y_pred_binary = [1 if value == cls else 0 for value in y_pred]
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0) * 100
        row = {
            "Endoscopic finding": config["class_names"][cls],
            "Samples": int(sum(y_true_binary)),
            "Accuracy (%)": recall,
            "Specificity (%)": binary_specificity(y_true_binary, y_pred_binary),
            "Precision (%)": precision_score(y_true_binary, y_pred_binary, zero_division=0) * 100,
            "Recall (%)": recall,
            "F1 score (%)": f1_score(y_true_binary, y_pred_binary, zero_division=0) * 100,
        }
        rows[str(row["Endoscopic finding"])] = row

    return [rows[name] for name in config["paper_order"]]


def build_targets_and_predictions(category: str, df: pd.DataFrame, dataset_root: Path) -> tuple[list[int], list[int]]:
    config = CATEGORY_CONFIG[category]
    truth_labels = load_truth_labels(dataset_root, category)
    y_true: list[int] = []
    y_pred: list[int] = []

    for _, row in df.iterrows():
        image = normalize_truth_key(row["image"])
        score = str(row["score"]).strip()
        truth = truth_labels.get(image)
        pred = config["score_to_label"].get(score)
        if truth is None or pred is None:
            continue
        y_true.append(int(config["truth_to_label"](truth)))
        y_pred.append(int(pred))

    return y_true, y_pred


def load_dlkgs_dataframe(result_root: Path, batch_folder: str) -> pd.DataFrame:
    workbook_path = result_root / batch_folder / "DLKGS" / "DLKGs_predictions.xlsx"
    df = pd.read_excel(workbook_path, sheet_name="predictions", usecols=["filepath", "score"]).copy()
    df = df.rename(columns={"filepath": "image"})
    df["image"] = df["image"].map(lambda value: normalize_truth_key(value).replace("/", "\\"))
    return df


def build_model_rows(
    result_root: Path,
    dataset_root: Path,
    batch_folder: str,
    model_configs: dict[str, str],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    batch_root = result_root / batch_folder

    for display_name, model_dir in model_configs.items():
        first_row_for_group = True
        for category in CATEGORIES:
            workbook_path = batch_root / model_dir / f"{category}_batch_result" / "result_batch.xlsx"
            df = pd.read_excel(workbook_path, usecols=["image", "score"])
            y_true, y_pred = build_targets_and_predictions(category, df, dataset_root)
            for metric_row in subclass_metric_rows(category, y_true, y_pred):
                rows.append({"Model/Group": display_name if first_row_for_group else "", **metric_row})
                first_row_for_group = False

    first_row_for_group = True
    dlkgs_df = load_dlkgs_dataframe(result_root, batch_folder)
    for category in CATEGORIES:
        subset = dlkgs_df[dlkgs_df["image"].astype(str).str.startswith(f"{category}\\")].copy()
        y_true, y_pred = build_targets_and_predictions(category, subset, dataset_root)
        for metric_row in subclass_metric_rows(category, y_true, y_pred):
            rows.append({"Model/Group": "DLKGs" if first_row_for_group else "", **metric_row})
            first_row_for_group = False

    return rows


def build_doctor_metric_rows(result_root: Path, dataset_root: Path) -> dict[str, dict[str, list[dict[str, float | int | str]]]]:
    doctor_root = result_root / "result_doctor"
    doctor_rows: dict[str, dict[str, list[dict[str, float | int | str]]]] = {}
    doctor_dirs = sorted(
        [path for path in doctor_root.iterdir() if path.is_dir() and path.name.startswith("doctor")],
        key=lambda path: int(path.name.replace("doctor", "")),
    )

    for doctor_dir in doctor_dirs:
        doctor_rows[doctor_dir.name] = {}
        for category in CATEGORIES:
            workbook_path = doctor_dir / f"{category}_result.xlsx"
            df = pd.read_excel(workbook_path, usecols=["image", "score"])
            y_true, y_pred = build_targets_and_predictions(category, df, dataset_root)
            doctor_rows[doctor_dir.name][category] = subclass_metric_rows(category, y_true, y_pred)

    return doctor_rows


def average_metric_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    return {
        "Endoscopic finding": rows[0]["Endoscopic finding"],
        "Samples": round(sum(int(row["Samples"]) for row in rows) / len(rows)),
        "Accuracy (%)": sum(float(row["Accuracy (%)"]) for row in rows) / len(rows),
        "Specificity (%)": sum(float(row["Specificity (%)"]) for row in rows) / len(rows),
        "Precision (%)": sum(float(row["Precision (%)"]) for row in rows) / len(rows),
        "Recall (%)": sum(float(row["Recall (%)"]) for row in rows) / len(rows),
        "F1 score (%)": sum(float(row["F1 score (%)"]) for row in rows) / len(rows),
    }


def build_doctor_group_rows(result_root: Path, dataset_root: Path) -> list[dict[str, float | int | str]]:
    doctor_rows = build_doctor_metric_rows(result_root, dataset_root)
    group_rows: dict[str, dict[str, list[dict[str, float | int | str]]]] = {}

    for group_name, doctors in DOCTOR_GROUPS.items():
        group_rows[group_name] = {}
        for category in CATEGORIES:
            group_rows[group_name][category] = []
            row_count = len(doctor_rows[doctors[0]][category])
            for row_index in range(row_count):
                group_rows[group_name][category].append(
                    average_metric_rows([doctor_rows[doctor][category][row_index] for doctor in doctors])
                )

    group_rows["Mixed"] = {}
    for category in CATEGORIES:
        group_rows["Mixed"][category] = []
        row_count = len(group_rows["Senior"][category])
        for row_index in range(row_count):
            group_rows["Mixed"][category].append(
                average_metric_rows([group_rows["Senior"][category][row_index], group_rows["Junior"][category][row_index]])
            )

    output_rows: list[dict[str, float | int | str]] = []
    for group_name in ["Senior", "Junior", "Mixed"]:
        first_row_for_group = True
        for category in CATEGORIES:
            for metric_row in group_rows[group_name][category]:
                output_rows.append({"Model/Group": group_name if first_row_for_group else "", **metric_row})
                first_row_for_group = False

    return output_rows


def build_subcategory_performance_table(
    result_root: Path,
    dataset_root: Path,
    batch_folder: str = "testdataset_result_batch",
    model_configs: dict[str, str] | None = None,
    include_doctor_groups: bool = True,
) -> pd.DataFrame:
    if model_configs is None:
        model_configs = TEST_MODEL_CONFIGS

    rows = build_model_rows(result_root, dataset_root, batch_folder, model_configs)
    if include_doctor_groups:
        rows.extend(build_doctor_group_rows(result_root, dataset_root))

    df = pd.DataFrame(rows)
    return df[
        [
            "Model/Group",
            "Endoscopic finding",
            "Accuracy (%)",
            "Specificity (%)",
            "Precision (%)",
            "Recall (%)",
            "F1 score (%)",
        ]
    ]


def format_table_for_txt(df: pd.DataFrame) -> str:
    display_df = df.copy()
    for column in ["Accuracy (%)", "Specificity (%)", "Precision (%)", "Recall (%)", "F1 score (%)"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.2f}")
    return display_df.to_string(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-subcategory performance metrics for models and doctor groups.")
    parser.add_argument("--dataset", choices=["test", "external"], default="test")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "external":
        dataset_root = args.dataset_root or PROJECT_ROOT / "dataset" / "externaldataset"
        output_prefix = args.output_prefix or "external_subcategory_performance_metrics"
        df = build_subcategory_performance_table(
            args.result_root,
            dataset_root,
            batch_folder="externaldataset_result_batch",
            model_configs=EXTERNAL_MODEL_CONFIGS,
            include_doctor_groups=False,
        )
    else:
        dataset_root = args.dataset_root or DEFAULT_DATASET_ROOT
        output_prefix = args.output_prefix or "subcategory_performance_metrics"
        df = build_subcategory_performance_table(args.result_root, dataset_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_xlsx = args.output_dir / f"{output_prefix}.xlsx"
    output_txt = args.output_dir / f"{output_prefix}.txt"

    df.to_excel(output_xlsx, index=False)
    output_txt.write_text(format_table_for_txt(df), encoding="utf-8")

    print(f"saved: {output_xlsx}")
    print(f"saved: {output_txt}")


if __name__ == "__main__":
    main()
