from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result" / "testdataset_result_batch"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "testdataset"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "image"

CATEGORIES = ["A", "DR", "H", "IM", "N"]
DATASET_CONFIGS = {
    "test": {
        "result_root": PROJECT_ROOT / "result" / "testdataset_result_batch",
        "dataset_root": PROJECT_ROOT / "dataset" / "testdataset",
        "matrix_folder": "matrix",
        "models": ["gemini", "gpt", "claude", "grok", "dlkgs"],
    },
    "external": {
        "result_root": PROJECT_ROOT / "result" / "externaldataset_result_batch",
        "dataset_root": PROJECT_ROOT / "dataset" / "externaldataset",
        "matrix_folder": "externalmatrix",
        "models": ["gemini", "dlkgs"],
    },
}
MODEL_CONFIGS = {
    "gemini": {"result_dirs": {"test": "result_gemini", "external": "gemini"}, "image_folder": "Gemini"},
    "gpt": {"result_dirs": {"test": "result_gpt"}, "image_folder": "GPT"},
    "claude": {"result_dirs": {"test": "result_claude"}, "image_folder": "Claude"},
    "grok": {"result_dirs": {"test": "result_grok"}, "image_folder": "Grok"},
    "dlkgs": {"result_dirs": {"test": "DLKGS", "external": "DLKGS"}, "image_folder": "DLKGS"},
}

CATEGORY_CONFIG = {
    "A": {
        "label_order": [3, 0, 1, 2],
        "display_labels": ["NA", "A-0", "A-1", "A-2"],
        "title": "Confusion matrix for A labels",
        "score_map": {
            "C-0": 0,
            "C-1": 0,
            "C-2": 1,
            "C-3": 1,
            "O-1": 2,
            "O-2": 2,
            "O-3": 2,
            "A0": 0,
            "A1": 1,
            "A2": 2,
            "NO": 3,
        },
        "no_label": 3,
    },
    "DR": {
        "label_order": [3, 0, 1, 2],
        "display_labels": ["NA", "DR-0", "DR-1", "DR-2"],
        "title": "Confusion matrix for DR labels",
        "score_map": {"DR-0": 0, "DR-1": 1, "DR-2": 2, "NO": 3},
        "no_label": 3,
    },
    "H": {
        "label_order": [2, 0, 1],
        "display_labels": ["NA", "H-0", "H-1"],
        "title": "Confusion matrix for H labels",
        "score_map": {"H-0": 0, "H-1": 1, "NO": 2},
        "no_label": 2,
    },
    "IM": {
        "label_order": [3, 0, 1, 2],
        "display_labels": ["NA", "IM-0", "IM-1", "IM-2"],
        "title": "Confusion matrix for IM labels",
        "score_map": {"IM-0": 0, "IM-1": 1, "IM-2": 2, "NO": 3},
        "no_label": 3,
    },
    "N": {
        "label_order": [2, 0, 1],
        "display_labels": ["NA", "N-0", "N-1"],
        "title": "Confusion matrix for N labels",
        "score_map": {"N-0": 0, "N-1": 1, "NO": 2, "no": 2},
        "no_label": 2,
    },
}

DLKGS_SCORE_MAP = {
    "A": {0: "NO", 1: "A0", 2: "A1", 3: "A2"},
    "DR": {0: "NO", 1: "DR-0", 2: "DR-1", 3: "DR-2"},
    "H": {0: "NO", 1: "H-0", 2: "H-1"},
    "IM": {0: "NO", 1: "IM-0", 2: "IM-1", 3: "IM-2"},
    "N": {0: "NO", 1: "N-0", 2: "N-1"},
}


def normalize_image(value: object) -> str:
    text = str(value).replace("\\", "/")
    return text.split(".")[0] + ".jpg"


def map_truth_label(category: str, value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() == "NO":
        return int(CATEGORY_CONFIG[category]["no_label"])
    try:
        return int(text)
    except ValueError:
        return None


def map_prediction_label(category: str, value: object) -> int | None:
    if value is None:
        return None
    return CATEGORY_CONFIG[category]["score_map"].get(str(value).strip())


def load_truth_labels(dataset_root: Path, category: str) -> dict[str, object]:
    path = dataset_root / category / "image_scores.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_llm_predictions(result_root: Path, dataset_key: str, model_key: str, category: str) -> pd.DataFrame:
    model_dir = MODEL_CONFIGS[model_key]["result_dirs"][dataset_key]
    workbook_path = result_root / model_dir / f"{category}_batch_result" / "result_batch.xlsx"
    return pd.read_excel(workbook_path, usecols=["image", "score"]).copy()


def dlkgs_image_value(item: dict) -> str:
    filepath = Path(item["filepath"])
    parts = filepath.parts
    lowered = [part.lower() for part in parts]
    if "testdataset" in lowered:
        idx = lowered.index("testdataset")
        return "\\".join(parts[idx + 1 :])
    return "\\".join([item["category"], item.get("sub_category", ""), item.get("filename", "")])


def load_dlkgs_predictions(result_root: Path, category: str) -> pd.DataFrame:
    json_path = result_root / "DLKGS" / "test_results.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    results = payload["results"] if isinstance(payload, dict) and "results" in payload else payload

    rows: list[dict[str, str]] = []
    for item in results:
        if item.get("category") != category:
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


def label_order_for_dataset(category: str, dataset_key: str) -> list[int]:
    labels = list(CATEGORY_CONFIG[category]["label_order"])
    if dataset_key != "external":
        return labels

    no_label = int(CATEGORY_CONFIG[category]["no_label"])
    return [label for label in labels if label != no_label] + [no_label]


def display_labels_for_dataset(category: str, dataset_key: str) -> list[str]:
    labels = list(CATEGORY_CONFIG[category]["label_order"])
    display_labels = list(CATEGORY_CONFIG[category]["display_labels"])
    if dataset_key != "external":
        return display_labels

    no_label = int(CATEGORY_CONFIG[category]["no_label"])
    label_to_display = dict(zip(labels, display_labels))
    ordered_labels = label_order_for_dataset(category, dataset_key)
    return [label_to_display[label] for label in ordered_labels]


def build_confusion_matrix(
    predictions: pd.DataFrame,
    truth_labels: dict[str, object],
    category: str,
    dataset_key: str,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    unmatched: list[tuple[str, str]] = []

    for _, row in predictions.iterrows():
        image = normalize_image(row["image"])
        truth = truth_labels.get(image)
        pred = map_prediction_label(category, row["score"])
        truth_mapped = map_truth_label(category, truth)

        if truth is None:
            unmatched.append((image, "missing ground truth"))
            continue
        if pred is None:
            unmatched.append((image, f"unmappable prediction: {row['score']}"))
            continue
        if truth_mapped is None:
            unmatched.append((image, f"unmappable truth: {truth}"))
            continue

        y_true.append(truth_mapped)
        y_pred.append(pred)

    labels = label_order_for_dataset(category, dataset_key)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, unmatched


def plot_confusion_matrix(cm: np.ndarray, category: str, dataset_key: str, output_path: Path) -> None:
    config = CATEGORY_CONFIG[category]
    display_labels = display_labels_for_dataset(category, dataset_key)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if i == j else "black"
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", color=color, fontsize=18)

    ax.set(
        xticks=np.arange(len(display_labels)),
        yticks=np.arange(len(display_labels)),
        xticklabels=display_labels,
        yticklabels=display_labels,
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(config["title"])

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    plt.colorbar(im, ax=ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def iter_model_keys(dataset_key: str, model_arg: str) -> list[str]:
    if model_arg == "all":
        return list(DATASET_CONFIGS[dataset_key]["models"])
    if model_arg not in DATASET_CONFIGS[dataset_key]["models"]:
        available = ", ".join(DATASET_CONFIGS[dataset_key]["models"])
        raise SystemExit(f"Model '{model_arg}' is not available for dataset '{dataset_key}'. Available: {available}")
    return [model_arg]


def iter_categories(category_arg: str) -> list[str]:
    if category_arg == "all":
        return CATEGORIES
    return [category_arg]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate confusion matrices for batch-result datasets.")
    parser.add_argument("--dataset", choices=DATASET_CONFIGS.keys(), default="test")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", choices=["all", *MODEL_CONFIGS.keys()], default="all")
    parser.add_argument("--category", choices=["all", *CATEGORIES], default="all")
    parser.add_argument("--matrix-folder", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    dataset_config = DATASET_CONFIGS[args.dataset]
    result_root = args.result_root or dataset_config["result_root"]
    dataset_root = args.dataset_root or dataset_config["dataset_root"]
    matrix_folder = args.matrix_folder or dataset_config["matrix_folder"]

    for model_key in iter_model_keys(args.dataset, args.model):
        image_folder = MODEL_CONFIGS[model_key]["image_folder"]
        for category in iter_categories(args.category):
            truth_labels = load_truth_labels(dataset_root, category)
            if model_key == "dlkgs":
                predictions = load_dlkgs_predictions(result_root, category)
            else:
                predictions = load_llm_predictions(result_root, args.dataset, model_key, category)

            cm, unmatched = build_confusion_matrix(predictions, truth_labels, category, args.dataset)
            output_path = args.output_root / image_folder / matrix_folder / f"confusion_matrix_{category}.png"
            plot_confusion_matrix(cm, category, args.dataset, output_path)
            print(f"saved: {output_path}")

            if unmatched:
                warnings.append(f"{image_folder}/{category}: {len(unmatched)} unmatched; first 3={unmatched[:3]}")

    for warning in warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
