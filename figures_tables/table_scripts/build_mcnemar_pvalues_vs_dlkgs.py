from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = Path(__file__).resolve().parents[1] / "table"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result" / "testdataset_result_batch"
DEFAULT_EXTERNAL_RESULT_ROOT = PROJECT_ROOT / "result" / "externaldataset_result_batch"
DEFAULT_OUTPUT = TABLE_DIR / "mcnemar_pvalues_vs_DLKGS.txt"
DEFAULT_EXTERNAL_OUTPUT = TABLE_DIR / "external_mcnemar_pvalues_Gemini_vs_DLKGs.txt"

REFERENCE_NAME = "result_DLKGS"
CATEGORIES = ["A", "DR", "H", "IM", "N"]
MODEL_DIRS = {
    "result_gemini": "result_gemini",
    "result_claude": "result_claude",
    "result_gpt": "result_gpt",
    "result_grok": "result_grok",
}

DLKGS_SCORE_MAP = {
    "A": {0: "NO", 1: "A0", 2: "A1", 3: "A2"},
    "DR": {0: "NO", 1: "DR-0", 2: "DR-1", 3: "DR-2"},
    "H": {0: "NO", 1: "H-0", 2: "H-1"},
    "IM": {0: "NO", 1: "IM-0", 2: "IM-1", 3: "IM-2"},
    "N": {0: "NO", 1: "N-0", 2: "N-1"},
}

EXTERNAL_CATEGORY_CONFIG = {
    "A": {
        "subgroups": ["A0", "A1", "A2", "NO"],
        "display": ["A-0", "A-1", "A-2", "NA"],
        "gemini_score": {"C-0": 0, "C-1": 0, "C-2": 1, "C-3": 1, "O-1": 2, "O-2": 2, "O-3": 2, "NO": 3},
        "dlkgs_score": {"A0": 0, "A1": 1, "A2": 2, "NO": 3},
    },
    "DR": {
        "subgroups": ["DR0", "DR1", "DR2", "NO"],
        "display": ["DR-0", "DR-1", "DR-2", "NA"],
        "gemini_score": {"DR-0": 0, "DR-1": 1, "DR-2": 2, "NO": 3},
        "dlkgs_score": {"DR0": 0, "DR1": 1, "DR2": 2, "NO": 3},
    },
    "H": {
        "subgroups": ["H0", "H1", "NO"],
        "display": ["H-0", "H-1", "NA"],
        "gemini_score": {"H-0": 0, "H-1": 1, "NO": 2},
        "dlkgs_score": {"H0": 0, "H1": 1, "NO": 2},
    },
    "IM": {
        "subgroups": ["IM0", "IM1", "IM2", "NO"],
        "display": ["IM-0", "IM-1", "IM-2", "NA"],
        "gemini_score": {"IM-0": 0, "IM-1": 1, "IM-2": 2, "NO": 3},
        "dlkgs_score": {"IM0": 0, "IM1": 1, "IM2": 2, "NO": 3},
    },
    "N": {
        "subgroups": ["N0", "N1", "NO"],
        "display": ["N-0", "N-1", "NA"],
        "gemini_score": {"N-0": 0, "N-1": 1, "NO": 2},
        "dlkgs_score": {"N0": 0, "N1": 1, "NO": 2},
    },
}

EXTERNAL_SUBGROUP_TO_INDEX = {
    category: {name: index for index, name in enumerate(config["subgroups"])}
    for category, config in EXTERNAL_CATEGORY_CONFIG.items()
}


def natural_key(text: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def subgroup_sort_key(name: str) -> tuple[object, ...]:
    category_order = {"A": 0, "DR": 1, "H": 2, "IM": 3, "N": 4}
    if "_" in name:
        category, subgroup = name.split("_", 1)
    else:
        category, subgroup = name, ""
    return (category_order.get(category, 999),) + natural_key(subgroup)


def extract_subgroup(image_value: str) -> str:
    normalized = str(image_value).replace("/", "\\")
    parts = normalized.split("\\")
    if len(parts) < 2:
        raise ValueError(f"Unexpected image path format: {image_value}")

    category = parts[0]
    subgroup = parts[1]
    if subgroup.lower() == "no":
        subgroup = "NO"
    return f"{category}_{subgroup}"


def natural_image_sort_key(image_value: object) -> tuple[tuple[object, ...], ...]:
    normalized = str(image_value).replace("/", "\\")
    return tuple(natural_key(part) for part in normalized.split("\\"))


def load_model_result(result_root: Path, model_dir: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for category in CATEGORIES:
        workbook_path = result_root / model_dir / f"{category}_batch_result" / "result_batch.xlsx"
        df = pd.read_excel(workbook_path, usecols=["image", "correct"]).copy()
        df["image"] = df["image"].astype(str).str.replace("/", "\\", regex=False)
        df["correct"] = df["correct"].astype(int)
        df["_image_sort_key"] = df["image"].map(natural_image_sort_key)
        df = df.sort_values("_image_sort_key", kind="stable").drop(columns="_image_sort_key")
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result["subgroup"] = result["image"].map(extract_subgroup)
    return result


def dlkgs_image_value(item: dict) -> str:
    filepath = Path(item["filepath"])
    parts = filepath.parts
    lowered = [part.lower() for part in parts]
    if "testdataset" in lowered:
        idx = lowered.index("testdataset")
        return "\\".join(parts[idx + 1 :])
    return "\\".join([item["category"], item.get("sub_category", ""), item.get("filename", "")])


def dlkgs_true_label(sub_category: str) -> int:
    subgroup = str(sub_category).upper()
    if subgroup == "NO":
        return 0

    match = re.search(r"\d+", subgroup)
    if match:
        return int(match.group()) + 1

    raise ValueError(f"Cannot parse DLKGs true label from sub_category={sub_category!r}")


def load_dlkgs_result(result_root: Path) -> pd.DataFrame:
    json_path = result_root / "DLKGS" / "test_results.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    results = payload["results"] if isinstance(payload, dict) and "results" in payload else payload

    rows: list[dict[str, str | int]] = []
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
                "correct": int(int(predicted_label) == dlkgs_true_label(item.get("sub_category", ""))),
            }
        )

    df = pd.DataFrame(rows)
    df["image"] = df["image"].astype(str).str.replace("/", "\\", regex=False)
    df["correct"] = df["correct"].astype(int)
    df["_category_order"] = df["image"].str.split("\\").str[0].map({category: i for i, category in enumerate(CATEGORIES)})
    df["_image_sort_key"] = df["image"].map(natural_image_sort_key)
    df = (
        df.sort_values(["_category_order", "_image_sort_key"], kind="stable")
        .drop(columns=["_category_order", "_image_sort_key", "score"])
        .reset_index(drop=True)
    )
    df["subgroup"] = df["image"].map(extract_subgroup)
    return df


def compute_mcnemar(correct_a: pd.Series, correct_b: pd.Series) -> tuple[int, int, int, int, float]:
    n00 = int(((correct_a == 1) & (correct_b == 1)).sum())
    n01 = int(((correct_a == 1) & (correct_b == 0)).sum())
    n10 = int(((correct_a == 0) & (correct_b == 1)).sum())
    n11 = int(((correct_a == 0) & (correct_b == 0)).sum())
    if n01 + n10 == 0:
        return n00, n01, n10, n11, 1.0

    result = mcnemar([[n00, n01], [n10, n11]], exact=False, correction=True)
    return n00, n01, n10, n11, float(result.pvalue)


def normalize_score(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip().upper()


def normalize_external_image(value: object, category_hint: str | None = None) -> str:
    text = "" if pd.isna(value) else str(value).strip().replace("/", "\\")
    parts = [part for part in text.split("\\") if part]
    categories = [category_hint] if category_hint else CATEGORIES
    category_index = None
    category = None
    for index, part in enumerate(parts):
        for candidate in categories:
            if candidate and part.upper() == candidate:
                category_index = index
                category = candidate
                break
        if category_index is not None:
            break

    if category_index is None:
        raise ValueError(f"Cannot locate category in path: {value}")

    parts = parts[category_index:]
    parts[0] = str(category)
    if len(parts) > 1:
        subgroup = parts[1].upper()
        if subgroup == "NO" or subgroup in EXTERNAL_CATEGORY_CONFIG[str(category)]["subgroups"]:
            parts[1] = subgroup
    if parts and "." in parts[-1]:
        parts[-1] = parts[-1].rsplit(".", 1)[0] + ".jpg"
    return "\\".join(parts)


def external_image_category(image: str) -> str:
    return image.split("\\")[0]


def external_image_subgroup(image: str) -> str:
    return image.split("\\")[1].upper()


def build_report_text(result_root: Path) -> str:
    reference_df = load_dlkgs_result(result_root).rename(columns={"correct": "correct_ref"})
    model_dfs = {
        name: load_model_result(result_root, model_dir).rename(columns={"correct": "correct_model"})
        for name, model_dir in MODEL_DIRS.items()
    }

    subgroup_order = sorted(reference_df["subgroup"].unique().tolist(), key=subgroup_sort_key)
    lines: list[str] = [
        "Two-sided McNemar P-values: each model vs result_DLKGS",
        "=" * 110,
        "",
    ]

    for model_name, model_df in model_dfs.items():
        merged = model_df.merge(
            reference_df[["image", "subgroup", "correct_ref"]],
            on=["image", "subgroup"],
            how="inner",
        )
        if len(merged) != len(reference_df):
            raise ValueError(
                f"Row alignment mismatch for {model_name}: merged={len(merged)} expected={len(reference_df)}"
            )

        lines.append(f"[{model_name} vs {REFERENCE_NAME}]")
        lines.append(f"{'Subgroup':<12}{'Samples':>8}{'n01':>8}{'n10':>8}{'P-value':>18}")

        for subgroup in subgroup_order:
            subset = merged[merged["subgroup"] == subgroup]
            n00, n01, n10, n11, p_value = compute_mcnemar(subset["correct_model"], subset["correct_ref"])
            lines.append(f"{subgroup:<12}{len(subset):>8}{n01:>8}{n10:>8}{p_value:>18.6e}")

        n00, n01, n10, n11, p_value = compute_mcnemar(merged["correct_model"], merged["correct_ref"])
        lines.append(f"{'overall':<12}{len(merged):>8}{n01:>8}{n10:>8}{p_value:>18.6e}")
        lines.append("")

    return "\n".join(lines) + "\n"


def load_external_gemini(result_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for category in CATEGORIES:
        workbook_path = result_root / "gemini" / f"{category}_batch_result" / "result_batch.xlsx"
        df = pd.read_excel(workbook_path, usecols=["image", "score"]).copy()
        df["image"] = df["image"].map(lambda value: normalize_external_image(value, category))
        df["category"] = category
        df["subgroup"] = df["image"].map(external_image_subgroup)
        df["true_label"] = df["subgroup"].map(EXTERNAL_SUBGROUP_TO_INDEX[category])
        df["pred_Gemini"] = df["score"].map(
            lambda value: EXTERNAL_CATEGORY_CONFIG[category]["gemini_score"].get(normalize_score(value))
        )
        if df[["true_label", "pred_Gemini"]].isna().any().any():
            bad = df[df[["true_label", "pred_Gemini"]].isna().any(axis=1)][["image", "score", "subgroup"]].head(10)
            raise ValueError(f"Unmapped Gemini rows for {category}: {bad.to_dict('records')}")
        df["correct_Gemini"] = (df["true_label"].astype(int) == df["pred_Gemini"].astype(int)).astype(int)
        frames.append(df[["image", "category", "subgroup", "true_label", "correct_Gemini"]])

    return pd.concat(frames, ignore_index=True)


def load_external_dlkgs(result_root: Path) -> pd.DataFrame:
    workbook_path = result_root / "DLKGS" / "DLKGs_predictions.xlsx"
    df = pd.read_excel(workbook_path, sheet_name="predictions", usecols=["filepath", "score", "groundtruth"]).copy()
    df = df.rename(columns={"filepath": "image"})
    df["image"] = df["image"].map(normalize_external_image)
    df["category"] = df["image"].map(external_image_category)
    df["subgroup"] = df["image"].map(external_image_subgroup)

    true_labels: list[int | None] = []
    pred_labels: list[int | None] = []
    for _, row in df.iterrows():
        category = row["category"]
        config = EXTERNAL_CATEGORY_CONFIG[category]
        true_labels.append(config["dlkgs_score"].get(normalize_score(row["groundtruth"])))
        pred_labels.append(config["dlkgs_score"].get(normalize_score(row["score"])))

    df["true_label_dlkgs"] = true_labels
    df["pred_DLKGS"] = pred_labels
    if df[["true_label_dlkgs", "pred_DLKGS"]].isna().any().any():
        bad = df[df[["true_label_dlkgs", "pred_DLKGS"]].isna().any(axis=1)][["image", "score", "groundtruth"]].head(10)
        raise ValueError(f"Unmapped DLKGS rows: {bad.to_dict('records')}")
    df["correct_DLKGS"] = (df["true_label_dlkgs"].astype(int) == df["pred_DLKGS"].astype(int)).astype(int)
    return df[["image", "category", "subgroup", "true_label_dlkgs", "correct_DLKGS"]]


def build_external_master(result_root: Path) -> pd.DataFrame:
    gemini = load_external_gemini(result_root)
    dlkgs = load_external_dlkgs(result_root)
    master = gemini.merge(dlkgs, on=["image", "category", "subgroup"], how="inner")
    if len(master) != len(gemini) or len(master) != len(dlkgs):
        raise ValueError(f"Image alignment mismatch: Gemini={len(gemini)}, DLKGS={len(dlkgs)}, merged={len(master)}")
    if not (master["true_label"] == master["true_label_dlkgs"]).all():
        bad = master[master["true_label"] != master["true_label_dlkgs"]][["image", "true_label", "true_label_dlkgs"]].head(10)
        raise ValueError(f"True-label mismatch: {bad.to_dict('records')}")
    return master.drop(columns=["true_label_dlkgs"])


def build_external_report_text(result_root: Path) -> str:
    master = build_external_master(result_root)
    rows: list[dict[str, object]] = []
    for category in CATEGORIES:
        config = EXTERNAL_CATEGORY_CONFIG[category]
        for subgroup, display in zip(config["subgroups"], config["display"]):
            subset = master[(master["category"] == category) & (master["subgroup"] == subgroup)]
            if subset.empty:
                continue
            n00, n01, n10, n11, p_value = compute_mcnemar(subset["correct_Gemini"], subset["correct_DLKGS"])
            rows.append({"label": f"{category}/{display}", "samples": len(subset), "n01": n01, "n10": n10, "p_value": p_value})

    lines = [
        "Two-sided McNemar P-values: Gemini vs DLKGs",
        "=" * 88,
        f"{'Subgroup':<14}{'Samples':>8}{'n01':>8}{'n10':>8}{'P-value':>18}",
    ]
    for row in rows:
        lines.append(
            f"{row['label']:<14}{row['samples']:>8}{row['n01']:>8}{row['n10']:>8}{row['p_value']:>18.6e}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build McNemar p-values for each MLLM vs DLKGs by subgroup.")
    parser.add_argument("--dataset", choices=["test", "external"], default="test")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "external":
        result_root = args.result_root or DEFAULT_EXTERNAL_RESULT_ROOT
        output = args.output or DEFAULT_EXTERNAL_OUTPUT
        report_text = build_external_report_text(result_root)
    else:
        result_root = args.result_root or DEFAULT_RESULT_ROOT
        output = args.output or DEFAULT_OUTPUT
        report_text = build_report_text(result_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report_text, encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
