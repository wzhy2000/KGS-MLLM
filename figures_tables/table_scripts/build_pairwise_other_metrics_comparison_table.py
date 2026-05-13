from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result" / "testdataset_result_batch"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "testdataset"
DEFAULT_EXTERNAL_RESULT_ROOT = PROJECT_ROOT / "result" / "externaldataset_result_batch"
DEFAULT_EXTERNAL_DATASET_ROOT = PROJECT_ROOT / "dataset" / "externaldataset"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "table"

CATEGORIES = ["A", "DR", "H", "IM", "N"]
MODEL_ORDER = ["Gemini", "GPT", "Claude", "Grok", "DLKGS"]
MODEL_DIRS = {
    "Gemini": "result_gemini",
    "GPT": "result_gpt",
    "Claude": "result_claude",
    "Grok": "result_grok",
}
EXTERNAL_MODEL_ORDER = ["Gemini", "DLKGS"]
EXTERNAL_MODEL_DIRS = {
    "Gemini": "gemini",
}
PAPER_MODEL_NAMES = {
    "Gemini": "KGS-Gemini 3",
    "GPT": "KGS-GPT 5.2",
    "Claude": "KGS-Claude 4.5",
    "Grok": "KGS-Grok 4",
    "DLKGS": "DLKGs",
}
METRIC_NAMES = ["specificity", "precision", "recall", "f1"]
METRIC_LABELS = {
    "specificity": "Specificity",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}
PAPER_METRIC_LABELS = {
    "Specificity": "Specificity (%)",
    "Precision": "Precision (%)",
    "Recall": "Recall (%)",
    "F1": "F1(%)",
}
N_BOOTSTRAPS = 2000
RANDOM_SEED = 1

CATEGORY_CONFIG = {
    "A": {
        "labels": [0, 1, 2, 3],
        "score_to_label": {"A0": 0, "A1": 1, "A2": 2, "NO": 3, "C-0": 0, "C-1": 0, "C-2": 1, "C-3": 1, "O-1": 2, "O-2": 2, "O-3": 2},
        "truth_to_label": lambda value: 3 if str(value).upper() == "NO" else int(value),
    },
    "DR": {
        "labels": [0, 1, 2, 3],
        "score_to_label": {"DR-0": 0, "DR-1": 1, "DR-2": 2, "DR0": 0, "DR1": 1, "DR2": 2, "NO": 3},
        "truth_to_label": lambda value: 3 if str(value).upper() == "NO" else int(value),
    },
    "H": {
        "labels": [0, 1, 2],
        "score_to_label": {"H-0": 0, "H-1": 1, "H0": 0, "H1": 1, "NO": 2},
        "truth_to_label": lambda value: 2 if str(value).upper() == "NO" else int(value),
    },
    "IM": {
        "labels": [0, 1, 2, 3],
        "score_to_label": {"IM-0": 0, "IM-1": 1, "IM-2": 2, "IM0": 0, "IM1": 1, "IM2": 2, "NO": 3},
        "truth_to_label": lambda value: 3 if str(value).upper() == "NO" else int(value),
    },
    "N": {
        "labels": [0, 1, 2],
        "score_to_label": {"N-0": 0, "N-1": 1, "N0": 0, "N1": 1, "NO": 2},
        "truth_to_label": lambda value: 2 if str(value).upper() == "NO" else int(value),
    },
}


def normalize_image(value: object) -> str:
    text = str(value).strip().replace("/", "\\")
    parts = [part for part in text.split("\\") if part]
    for index, part in enumerate(parts):
        if part in CATEGORIES:
            return "\\".join(parts[index:])
    return text


def truth_image_key(value: object) -> str:
    return normalize_image(value).replace("\\", "/")


def natural_key(text: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def natural_image_sort_key(image_value: object) -> tuple[tuple[object, ...], ...]:
    return tuple(natural_key(part) for part in normalize_image(image_value).split("\\"))


def extract_category(image_value: object) -> str:
    return normalize_image(image_value).split("\\")[0]


def load_truth_maps(dataset_root: Path) -> dict[str, dict[str, object]]:
    return {
        category: json.loads((dataset_root / category / "image_scores.json").read_text(encoding="utf-8"))
        for category in CATEGORIES
    }


def load_mllm_predictions(
    result_root: Path,
    model_name: str,
    model_dirs: dict[str, str] | None = None,
    sort_predictions: bool = True,
) -> pd.DataFrame:
    model_dirs = model_dirs or MODEL_DIRS
    frames: list[pd.DataFrame] = []
    model_root = result_root / model_dirs[model_name]
    for category in CATEGORIES:
        workbook_path = model_root / f"{category}_batch_result" / "result_batch.xlsx"
        df = pd.read_excel(workbook_path, usecols=["image", "score"]).copy()
        df["image"] = df["image"].map(normalize_image)
        df["category"] = category
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    if not sort_predictions:
        return result.reset_index(drop=True)
    result["_image_sort_key"] = result["image"].map(natural_image_sort_key)
    return result.sort_values("_image_sort_key", kind="stable").drop(columns="_image_sort_key").reset_index(drop=True)


def load_dlkgs_predictions(result_root: Path, sort_predictions: bool = True) -> pd.DataFrame:
    workbook_path = result_root / "DLKGS" / "DLKGs_predictions.xlsx"
    df = pd.read_excel(workbook_path, sheet_name="predictions", usecols=["filepath", "score"]).copy()
    df = df.rename(columns={"filepath": "image"})
    df["image"] = df["image"].map(normalize_image)
    df["category"] = df["image"].map(extract_category)
    if not sort_predictions:
        return df.reset_index(drop=True)
    df["_image_sort_key"] = df["image"].map(natural_image_sort_key)
    return df.sort_values("_image_sort_key", kind="stable").drop(columns="_image_sort_key").reset_index(drop=True)


def build_master_dataframe(
    result_root: Path,
    dataset_root: Path,
    model_order: list[str] | None = None,
    model_dirs: dict[str, str] | None = None,
    sort_predictions: bool = True,
) -> pd.DataFrame:
    model_order = model_order or MODEL_ORDER
    model_dirs = model_dirs or MODEL_DIRS
    truth_maps = load_truth_maps(dataset_root)
    reference_model = next(model_name for model_name in model_order if model_name != "DLKGS")
    reference = load_mllm_predictions(result_root, reference_model, model_dirs, sort_predictions)[["image", "category"]].copy()

    true_labels: list[int] = []
    for image_value, category in zip(reference["image"], reference["category"]):
        truth_value = truth_maps[category][truth_image_key(image_value)]
        true_labels.append(CATEGORY_CONFIG[category]["truth_to_label"](truth_value))
    reference["true_label"] = np.array(true_labels, dtype=int)

    master = reference.copy()
    prediction_frames = {
        model_name: load_mllm_predictions(result_root, model_name, model_dirs, sort_predictions)
        for model_name in model_order
        if model_name != "DLKGS"
    }
    if "DLKGS" in model_order:
        prediction_frames["DLKGS"] = load_dlkgs_predictions(result_root, sort_predictions)

    for model_name in model_order:
        df = prediction_frames[model_name].copy()
        pred_labels: list[int] = []
        for score, category in zip(df["score"], df["category"]):
            pred_labels.append(CATEGORY_CONFIG[category]["score_to_label"][str(score).strip()])
        df[f"pred_{model_name}"] = np.array(pred_labels, dtype=int)
        master = master.merge(df[["image", f"pred_{model_name}"]], on="image", how="inner")

    if len(master) != len(reference):
        raise ValueError(f"Image alignment mismatch: reference={len(reference)}, merged={len(master)}")

    return master


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_count: int) -> dict[str, float]:
    cm = np.bincount(label_count * y_true + y_pred, minlength=label_count * label_count).reshape(label_count, label_count)
    total = cm.sum()
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    tn = float(total) - tp - fp - fn

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

    return {
        "specificity": float(specificity.mean()) if len(specificity) else 0.0,
        "precision": float(precision.mean()) if len(precision) else 0.0,
        "recall": float(recall.mean()) if len(recall) else 0.0,
        "f1": float(f1.mean()) if len(f1) else 0.0,
    }


def prepare_category_arrays(master: pd.DataFrame, model_order: list[str] | None = None) -> dict[str, dict[str, np.ndarray]]:
    model_order = model_order or MODEL_ORDER
    prepared: dict[str, dict[str, np.ndarray]] = {}
    for category in CATEGORIES:
        subset = master[master["category"] == category]
        payload = {"true": subset["true_label"].to_numpy(dtype=int)}
        for model_name in model_order:
            payload[f"pred_{model_name}"] = subset[f"pred_{model_name}"].to_numpy(dtype=int)
        prepared[category] = payload
    return prepared


def weighted_overall_metrics(master: pd.DataFrame, model_name: str) -> dict[str, float]:
    weighted_sum = {metric: 0.0 for metric in METRIC_NAMES}
    total_samples = len(master)
    for category in CATEGORIES:
        subset = master[master["category"] == category]
        metrics = multiclass_metrics(
            subset["true_label"].to_numpy(dtype=int),
            subset[f"pred_{model_name}"].to_numpy(dtype=int),
            len(CATEGORY_CONFIG[category]["labels"]),
        )
        for metric in METRIC_NAMES:
            weighted_sum[metric] += metrics[metric] * len(subset)

    return {metric: value / total_samples for metric, value in weighted_sum.items()}


def run_bootstrap_other_metrics(
    master: pd.DataFrame,
    n_bootstraps: int = N_BOOTSTRAPS,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    model_order = model_order or MODEL_ORDER
    prepared = prepare_category_arrays(master, model_order)
    observed = {model_name: weighted_overall_metrics(master, model_name) for model_name in model_order}
    rng = np.random.default_rng(RANDOM_SEED)

    boot = {model_name: {metric: np.zeros(n_bootstraps, dtype=float) for metric in METRIC_NAMES} for model_name in model_order}
    total_samples = sum(len(prepared[category]["true"]) for category in CATEGORIES)

    for bootstrap_index in range(n_bootstraps):
        for model_name in model_order:
            weighted_sum = {metric: 0.0 for metric in METRIC_NAMES}
            for category in CATEGORIES:
                true_arr = prepared[category]["true"]
                pred_arr = prepared[category][f"pred_{model_name}"]
                n_category = len(true_arr)
                indices = rng.integers(0, n_category, size=n_category)
                metrics = multiclass_metrics(
                    true_arr[indices],
                    pred_arr[indices],
                    len(CATEGORY_CONFIG[category]["labels"]),
                )
                for metric in METRIC_NAMES:
                    weighted_sum[metric] += metrics[metric] * n_category

            for metric in METRIC_NAMES:
                boot[model_name][metric][bootstrap_index] = weighted_sum[metric] / total_samples

    rows: list[dict[str, float | int | str]] = []
    for metric in METRIC_NAMES:
        for model_a, model_b in combinations(model_order, 2):
            diffs = boot[model_a][metric] - boot[model_b][metric]
            p_lower = float(np.mean(diffs <= 0))
            p_upper = float(np.mean(diffs >= 0))
            p_value = min(1.0, max(1.0 / (n_bootstraps + 1), 2 * min(p_lower, p_upper)))
            rows.append(
                {
                    "Indicator": METRIC_LABELS[metric],
                    "Comparison": f"{model_a} vs {model_b}",
                    "Samples": len(master),
                    "Model A": model_a,
                    "Model B": model_b,
                    "Mean Diff": observed[model_a][metric] - observed[model_b][metric],
                    "Bootstrap P value": p_value,
                    "95% CI Lower": float(np.percentile(diffs, 2.5)),
                    "95% CI Upper": float(np.percentile(diffs, 97.5)),
                }
            )

    return pd.DataFrame(rows)


def p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def external_p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value <= 0.05:
        return "*"
    return ""


def paper_comparison_name(comparison: str) -> str:
    left, right = comparison.split(" vs ")
    return f"{PAPER_MODEL_NAMES[left]} vs {PAPER_MODEL_NAMES[right]}"


def format_bootstrap_p_value(value: float) -> str:
    return "<0.0005" if value <= 1.0 / (N_BOOTSTRAPS + 1) else f"{value:.4f}"


def build_paper_table(results_df: pd.DataFrame) -> pd.DataFrame:
    output_frames: list[pd.DataFrame] = []
    for indicator, metric_df in results_df.groupby("Indicator", sort=False):
        _, adjusted_values, _, _ = multipletests(
            metric_df["Bootstrap P value"].to_numpy(dtype=float),
            alpha=0.05,
            method="bonferroni",
        )
        metric_rows: list[dict[str, float | int | str]] = []
        for (_, row), adjusted_p in zip(metric_df.iterrows(), adjusted_values):
            metric_rows.append(
                {
                    "Performance metric": PAPER_METRIC_LABELS[str(row["Indicator"])],
                    "Comparison": paper_comparison_name(str(row["Comparison"])),
                    "Samples": int(row["Samples"]),
                    "Model A": PAPER_MODEL_NAMES[str(row["Model A"])],
                    "Model B": PAPER_MODEL_NAMES[str(row["Model B"])],
                    "Mean difference (%)": round(float(row["Mean Diff"]) * 100, 2),
                    "95% CI lower (%)": round(float(row["95% CI Lower"]) * 100, 2),
                    "95% CI upper (%)": round(float(row["95% CI Upper"]) * 100, 2),
                    "Bootstrap p value": f"{format_bootstrap_p_value(float(row['Bootstrap P value']))}{p_to_stars(float(row['Bootstrap P value']))}",
                }
            )
        output_frames.append(pd.DataFrame(metric_rows))

    return pd.concat(output_frames, ignore_index=True)


def format_external_p_value(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def build_external_paper_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for _, row in results_df.iterrows():
        rows.append(
            {
                "Performance metric": PAPER_METRIC_LABELS[str(row["Indicator"])],
                "Comparison": paper_comparison_name(str(row["Comparison"])),
                "Samples": int(row["Samples"]),
                "Model A": PAPER_MODEL_NAMES[str(row["Model A"])],
                "Model B": PAPER_MODEL_NAMES[str(row["Model B"])],
                "Mean difference (%)": round(float(row["Mean Diff"]) * 100, 4),
                "95% CI lower (%)": round(float(row["95% CI Lower"]) * 100, 4),
                "95% CI upper (%)": round(float(row["95% CI Upper"]) * 100, 4),
                "Bootstrap p value": f"{format_external_p_value(float(row['Bootstrap P value']))}{external_p_to_stars(float(row['Bootstrap P value']))}",
            }
        )
    return pd.DataFrame(rows)


def format_paper_table_for_txt(paper_df: pd.DataFrame) -> str:
    display_df = paper_df.copy()
    for column in ["Mean difference (%)", "95% CI lower (%)", "95% CI upper (%)"]:
        if column in display_df.columns:
            precision = 4 if len(display_df) == 4 else 2
            display_df[column] = display_df[column].map(lambda value: f"{value:.{precision}f}")
    return display_df.to_string(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pairwise specificity, precision, recall, and F1 comparison table for the KGS test set."
    )
    parser.add_argument("--dataset", choices=["test", "external"], default="test")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--n-bootstraps", type=int, default=N_BOOTSTRAPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "external":
        result_root = args.result_root or DEFAULT_EXTERNAL_RESULT_ROOT
        dataset_root = args.dataset_root or DEFAULT_EXTERNAL_DATASET_ROOT
        output_prefix = args.output_prefix or "external_other_metrics_comparison_results"
        master = build_master_dataframe(
            result_root,
            dataset_root,
            EXTERNAL_MODEL_ORDER,
            EXTERNAL_MODEL_DIRS,
            sort_predictions=False,
        )
        results_df = run_bootstrap_other_metrics(master, args.n_bootstraps, EXTERNAL_MODEL_ORDER)
        output_df = build_external_paper_table(results_df)
    else:
        result_root = args.result_root or DEFAULT_RESULT_ROOT
        dataset_root = args.dataset_root or DEFAULT_DATASET_ROOT
        output_prefix = args.output_prefix or "other_metrics_comparison_results"
        master = build_master_dataframe(result_root, dataset_root)
        results_df = run_bootstrap_other_metrics(master, args.n_bootstraps)
        output_df = build_paper_table(results_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_xlsx = args.output_dir / f"{output_prefix}.xlsx"
    output_txt = args.output_dir / f"{output_prefix}.txt"

    output_df.to_excel(output_xlsx, index=False)
    output_txt.write_text(format_paper_table_for_txt(output_df), encoding="utf-8")

    print(f"saved: {output_xlsx}")
    print(f"saved: {output_txt}")


if __name__ == "__main__":
    main()
