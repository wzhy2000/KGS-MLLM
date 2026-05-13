from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result" / "testdataset_result_batch"
DEFAULT_EXTERNAL_RESULT_ROOT = PROJECT_ROOT / "result" / "externaldataset_result_batch"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "table"

CATEGORIES = ["A", "DR", "H", "IM", "N"]
MODEL_ORDER = ["Gemini", "GPT", "Claude", "Grok", "DLKGS"]
MODEL_DIRS = {
    "Gemini": "result_gemini",
    "GPT": "result_gpt",
    "Claude": "result_claude",
    "Grok": "result_grok",
}
PAPER_MODEL_NAMES = {
    "Gemini": "KGS-Gemini 3",
    "GPT": "KGS-GPT 5.2",
    "Claude": "KGS-Claude 4.5",
    "Grok": "KGS-Grok 4",
    "DLKGS": "DLKGs",
}


def normalize_image(value: object) -> str:
    return str(value).strip().replace("/", "\\")


def normalize_relative_image(value: object) -> str:
    text = normalize_image(value)
    parts = [part for part in text.split("\\") if part]
    for index, part in enumerate(parts):
        if part in CATEGORIES:
            return "\\".join(parts[index:])
    return text


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


def compute_bootstrap_ci(
    y1: np.ndarray,
    y2: np.ndarray,
    n_bootstraps: int = 10000,
    ci: int = 95,
) -> tuple[float, float]:
    rng = np.random.default_rng(1)
    n = len(y1)
    indices = rng.integers(0, n, size=(n_bootstraps, n))
    diffs = np.mean(y1[indices], axis=1) - np.mean(y2[indices], axis=1)
    alpha = (100 - ci) / 2
    return float(np.percentile(diffs, alpha)), float(np.percentile(diffs, 100 - alpha))


def load_mllm_results(result_root: Path, model_name: str) -> pd.DataFrame:
    model_dir = result_root / MODEL_DIRS[model_name]
    frames: list[pd.DataFrame] = []
    for category in CATEGORIES:
        workbook_path = model_dir / f"{category}_batch_result" / "result_batch.xlsx"
        df = pd.read_excel(workbook_path, usecols=["image", "correct"]).copy()
        df["image"] = df["image"].map(normalize_image)
        df["correct"] = df["correct"].astype(int)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result["_image_sort_key"] = result["image"].map(natural_image_sort_key)
    return result.sort_values("_image_sort_key", kind="stable").drop(columns="_image_sort_key").reset_index(drop=True)


def load_external_gemini_results(result_root: Path) -> pd.DataFrame:
    model_dir = result_root / "gemini"
    frames: list[pd.DataFrame] = []
    for category in CATEGORIES:
        workbook_path = model_dir / f"{category}_batch_result" / "result_batch.xlsx"
        df = pd.read_excel(workbook_path, usecols=["image", "correct"]).copy()
        df["image"] = df["image"].map(normalize_relative_image)
        df["correct"] = df["correct"].astype(int)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    return result.reset_index(drop=True)


def load_dlkgs_results(result_root: Path) -> pd.DataFrame:
    workbook_path = result_root / "DLKGS" / "DLKGs_predictions.xlsx"
    df = pd.read_excel(workbook_path, sheet_name="predictions", usecols=["filepath", "is_correct"]).copy()
    df = df.rename(columns={"filepath": "image", "is_correct": "correct"})
    df["image"] = df["image"].map(normalize_image)
    df["correct"] = df["correct"].astype(int)
    df["_image_sort_key"] = df["image"].map(natural_image_sort_key)
    df = df.sort_values("_image_sort_key", kind="stable").drop(columns="_image_sort_key").reset_index(drop=True)
    return df[["image", "correct"]]


def load_external_dlkgs_results(result_root: Path) -> pd.DataFrame:
    workbook_path = result_root / "DLKGS" / "DLKGs_predictions.xlsx"
    df = pd.read_excel(workbook_path, sheet_name="predictions", usecols=["filepath", "is_correct"]).copy()
    df = df.rename(columns={"filepath": "image", "is_correct": "correct"})
    df["image"] = df["image"].map(normalize_relative_image)
    df["correct"] = df["correct"].astype(int)
    return df[["image", "correct"]].reset_index(drop=True)


def load_all_model_results(result_root: Path) -> dict[str, pd.DataFrame]:
    loaded = {model_name: load_mllm_results(result_root, model_name) for model_name in MODEL_ORDER if model_name != "DLKGS"}
    loaded["DLKGS"] = load_dlkgs_results(result_root)
    return loaded


def load_external_model_results(result_root: Path) -> dict[str, pd.DataFrame]:
    return {
        "Gemini": load_external_gemini_results(result_root),
        "DLKGS": load_external_dlkgs_results(result_root),
    }


def merge_model_pair(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> pd.DataFrame:
    merged = left.merge(right, on="image", suffixes=(f"_{left_name}", f"_{right_name}"), how="inner")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            f"Image alignment mismatch for {left_name} vs {right_name}: "
            f"left={len(left)}, right={len(right)}, merged={len(merged)}"
        )
    return merged


def compute_mcnemar_pvalue(y1: np.ndarray, y2: np.ndarray) -> tuple[int, int, int, int, float]:
    n00 = int(np.sum((y1 == 1) & (y2 == 1)))
    n01 = int(np.sum((y1 == 1) & (y2 == 0)))
    n10 = int(np.sum((y1 == 0) & (y2 == 1)))
    n11 = int(np.sum((y1 == 0) & (y2 == 0)))
    if n01 + n10 == 0:
        return n00, n01, n10, n11, 1.0

    result = mcnemar([[n00, n01], [n10, n11]], exact=False, correction=True)
    return n00, n01, n10, n11, float(result.pvalue)


def build_pairwise_accuracy_comparison(
    result_root: Path,
    adjustment: str = "fdr_bh",
    model_order: list[str] | None = None,
    loaded: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    model_order = model_order or MODEL_ORDER
    loaded = loaded or load_all_model_results(result_root)
    rows: list[dict[str, float | int | str]] = []

    for model_a, model_b in combinations(model_order, 2):
        merged = merge_model_pair(loaded[model_a], loaded[model_b], model_a, model_b)
        data_a = merged[f"correct_{model_a}"].to_numpy(dtype=int)
        data_b = merged[f"correct_{model_b}"].to_numpy(dtype=int)

        acc_a = float(np.mean(data_a == 1))
        acc_b = float(np.mean(data_b == 1))
        ci_lower, ci_upper = compute_bootstrap_ci(data_a, data_b)
        n00, n01, n10, n11, p_value = compute_mcnemar_pvalue(data_a, data_b)

        rows.append(
            {
                "Comparison": f"{model_a} vs {model_b}",
                "Samples": len(data_a),
                "Accuracy A": acc_a,
                "Accuracy B": acc_b,
                "Mean Diff (Acc)": acc_a - acc_b,
                "Bootstrap Lower 95%CI": ci_lower,
                "Bootstrap Upper 95%CI": ci_upper,
                "n00": n00,
                "n01": n01,
                "n10": n10,
                "n11": n11,
                "P-value (McNemar)": p_value,
            }
        )

    results_df = pd.DataFrame(rows)
    _, corrected, _, _ = multipletests(
        results_df["P-value (McNemar)"].to_numpy(dtype=float),
        alpha=0.05,
        method=adjustment,
    )
    column_name = "FDR Corrected P" if adjustment == "fdr_bh" else "Bonferroni Corrected P"
    results_df[column_name] = corrected
    return results_df


def build_external_accuracy_comparison(result_root: Path) -> pd.DataFrame:
    return build_pairwise_accuracy_comparison(
        result_root,
        adjustment="bonferroni",
        model_order=["Gemini", "DLKGS"],
        loaded=load_external_model_results(result_root),
    )


def p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def format_scientific_for_paper(value: float) -> str:
    mantissa_text, exponent_text = f"{value:.2e}".split("e")
    exponent = int(exponent_text)
    return f"{mantissa_text} x 10^{exponent}"


def paper_comparison_name(comparison: str) -> str:
    left, right = comparison.split(" vs ")
    return f"{PAPER_MODEL_NAMES[left]} vs {PAPER_MODEL_NAMES[right]}"


def build_paper_table(results_df: pd.DataFrame) -> pd.DataFrame:
    raw_pvalues = results_df["P-value (McNemar)"].to_numpy(dtype=float)
    _, bonferroni_pvalues, _, _ = multipletests(raw_pvalues, alpha=0.05, method="bonferroni")

    rows: list[dict[str, str | float]] = []
    for (_, row), adjusted_p in zip(results_df.iterrows(), bonferroni_pvalues):
        rows.append(
            {
                "Comparison": paper_comparison_name(str(row["Comparison"])),
                "Samples": int(row["Samples"]),
                "Accuracy A": float(row["Accuracy A"]),
                "Accuracy B": float(row["Accuracy B"]),
                "Accuracy difference (%)": round(float(row["Mean Diff (Acc)"]) * 100, 2),
                "95% CI lower (%)": round(float(row["Bootstrap Lower 95%CI"]) * 100, 2),
                "95% CI upper (%)": round(float(row["Bootstrap Upper 95%CI"]) * 100, 2),
                "n00": int(row["n00"]),
                "n01": int(row["n01"]),
                "n10": int(row["n10"]),
                "n11": int(row["n11"]),
                "p value (McNemar)": format_scientific_for_paper(float(row["P-value (McNemar)"])),
                "Adjusted p value": f"{format_scientific_for_paper(float(adjusted_p))}{p_to_stars(float(adjusted_p))}",
            }
        )

    return pd.DataFrame(rows)


def build_external_paper_table(results_df: pd.DataFrame) -> pd.DataFrame:
    return build_paper_table(results_df)


def format_paper_table_for_txt(paper_df: pd.DataFrame) -> str:
    display_df = paper_df.copy()
    for column in ["Accuracy A", "Accuracy B"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: f"{value:.5f}")
    for column in ["Accuracy difference (%)", "95% CI lower (%)", "95% CI upper (%)", "95% CI lower", "95% CI upper"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: f"{value:.2f}")
    return display_df.to_string(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pairwise image-level accuracy comparison table for the KGS test set.")
    parser.add_argument("--dataset", choices=["test", "external"], default="test")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--adjustment", choices=["fdr_bh", "bonferroni"], default="fdr_bh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "external":
        result_root = args.result_root or DEFAULT_EXTERNAL_RESULT_ROOT
        output_prefix = args.output_prefix or "external_statistical_comparison_results"
        results_df = build_external_accuracy_comparison(result_root)
        output_df = build_external_paper_table(results_df)
    else:
        result_root = args.result_root or DEFAULT_RESULT_ROOT
        output_prefix = args.output_prefix or "statistical_comparison_results"
        results_df = build_pairwise_accuracy_comparison(result_root, args.adjustment)
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
