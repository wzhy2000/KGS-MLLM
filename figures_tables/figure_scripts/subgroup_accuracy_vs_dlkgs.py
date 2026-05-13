from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLE_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "table_scripts"
if str(TABLE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(TABLE_SCRIPT_DIR))

from build_mcnemar_pvalues_vs_dlkgs import DEFAULT_EXTERNAL_RESULT_ROOT, EXTERNAL_CATEGORY_CONFIG, build_external_master, compute_mcnemar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "result" / "testdataset_result_batch"
TABLE_DIR = Path(__file__).resolve().parents[1] / "table"
IMAGE_DIR = Path(__file__).resolve().parents[1] / "image"

P_VALUE_PATH = TABLE_DIR / "mcnemar_pvalues_vs_DLKGS.txt"

CATEGORIES = ["A", "DR", "H", "IM", "N"]
CATEGORY_ORDER = {
    "A": ["A-N", "A0", "A1", "A2"],
    "DR": ["DR-N", "DR0", "DR1", "DR2"],
    "H": ["H-N", "H0", "H1"],
    "IM": ["IM-N", "IM0", "IM1", "IM2"],
    "N": ["N-N", "N0", "N1"],
}

MODEL_CONFIGS = {
    "gemini": {
        "display": "Gemini3",
        "block": "result_gemini",
        "result_dir": "result_gemini",
        "folder": "Gemini",
        "output": "accuracy_comparison_Gemini3.png",
    },
    "gpt": {
        "display": "GPT5.2",
        "block": "result_gpt",
        "result_dir": "result_gpt",
        "folder": "GPT",
        "output": "accuracy_comparison_GPT5.2.png",
    },
    "claude": {
        "display": "Claude4.5",
        "block": "result_claude",
        "result_dir": "result_claude",
        "folder": "Claude",
        "output": "accuracy_comparison_Claude4.5.png",
    },
    "grok": {
        "display": "Grok4",
        "block": "result_grok",
        "result_dir": "result_grok",
        "folder": "Grok",
        "output": "accuracy_comparison_Grok4.png",
    },
}

DLKGS_SCORE_MAP = {
    "A": {0: "NO", 1: "A0", 2: "A1", 3: "A2"},
    "DR": {0: "NO", 1: "DR-0", 2: "DR-1", 3: "DR-2"},
    "H": {0: "NO", 1: "H-0", 2: "H-1"},
    "IM": {0: "NO", 1: "IM-0", 2: "IM-1", 3: "IM-2"},
    "N": {0: "NO", 1: "N-0", 2: "N-1"},
}

P_VALUE_SUBGROUP_MAP = {
    "A_A0": "A0",
    "A_A1": "A1",
    "A_A2": "A2",
    "A_NO": "A-N",
    "DR_DR0": "DR0",
    "DR_DR1": "DR1",
    "DR_DR2": "DR2",
    "DR_NO": "DR-N",
    "H_H0": "H0",
    "H_H1": "H1",
    "H_NO": "H-N",
    "IM_IM0": "IM0",
    "IM_IM1": "IM1",
    "IM_IM2": "IM2",
    "IM_NO": "IM-N",
    "N_N0": "N0",
    "N_N1": "N1",
    "N_NO": "N-N",
}


def subgroup_to_display(category: str, subgroup: str) -> str:
    subgroup_upper = subgroup.upper()
    if subgroup_upper == "NO":
        return f"{category}-N"
    return subgroup_upper


def get_dlkgs_true_label(sub_category: str) -> int:
    subgroup = str(sub_category).upper()
    if subgroup == "NO":
        return 0

    match = re.search(r"\d+", subgroup)
    if match:
        return int(match.group()) + 1

    raise ValueError(f"Cannot parse DLKGs true label from sub_category={sub_category!r}")


def dlkgs_image_value(item: dict) -> str:
    filepath = Path(item["filepath"])
    parts = filepath.parts
    lowered = [part.lower() for part in parts]
    if "testdataset" in lowered:
        idx = lowered.index("testdataset")
        return "\\".join(parts[idx + 1 :])
    return "\\".join([item["category"], item.get("sub_category", ""), item.get("filename", "")])


def load_model_dataframe(model_key: str, result_root: Path) -> pd.DataFrame:
    config = MODEL_CONFIGS[model_key]
    model_dir = result_root / config["result_dir"]
    frames: list[pd.DataFrame] = []

    for category in CATEGORIES:
        workbook_path = model_dir / f"{category}_batch_result" / "result_batch.xlsx"
        df = pd.read_excel(workbook_path, usecols=["image", "correct"]).copy()
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_dlkgs_dataframe(result_root: Path) -> pd.DataFrame:
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

        true_label = get_dlkgs_true_label(item.get("sub_category", ""))
        rows.append(
            {
                "image": dlkgs_image_value(item),
                "correct": int(int(predicted_label) == true_label),
                "score": DLKGS_SCORE_MAP[category][int(predicted_label)],
            }
        )

    return pd.DataFrame(rows)


def build_accuracy_dict(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    prepared = df[["image", "correct"]].copy()
    prepared["image"] = prepared["image"].astype(str).str.replace("/", "\\", regex=False)
    prepared["correct"] = prepared["correct"].astype(float)

    grouped: dict[str, dict[str, float]] = {category: {} for category in CATEGORIES}
    for category in grouped:
        subset = prepared[prepared["image"].str.startswith(f"{category}\\")].copy()
        subset["subgroup"] = subset["image"].str.split("\\").str[1]
        for subgroup, subgroup_df in subset.groupby("subgroup", sort=False):
            display_name = subgroup_to_display(category, subgroup)
            grouped[category][display_name] = float(subgroup_df["correct"].mean())

    return {
        category: {subgroup: grouped[category][subgroup] for subgroup in CATEGORY_ORDER[category]}
        for category in CATEGORIES
    }


def build_pvalue_dict(txt_path: Path, block_name: str) -> dict[str, float]:
    text = txt_path.read_text(encoding="utf-8")
    current_block: str | None = None
    pvals: dict[str, float] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            current_block = line[1:-1]
            continue

        if current_block != f"{block_name} vs result_DLKGS":
            continue

        if line.startswith("Subgroup") or line.startswith("=") or line.startswith("overall"):
            continue

        parts = re.split(r"\s+", line)
        subgroup = parts[0]
        if subgroup in P_VALUE_SUBGROUP_MAP:
            pvals[P_VALUE_SUBGROUP_MAP[subgroup]] = float(parts[4])

    return pvals


def p_to_star(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_model_vs_dlkgs(model_key: str, output_path: Path, result_root: Path, p_value_path: Path) -> None:
    config = MODEL_CONFIGS[model_key]
    model_name = config["display"]

    model_data = build_accuracy_dict(load_model_dataframe(model_key, result_root))
    dlkgs_data = build_accuracy_dict(load_dlkgs_dataframe(result_root))
    p_values = build_pvalue_dict(p_value_path, config["block"])

    color_palette = ["#4DBBD5", "#E64B35", "#00A087", "#3C5488"]
    labels: list[str] = []
    values_left: list[float] = []
    values_right: list[float] = []
    bar_colors: list[str] = []
    positions_left: list[float] = []
    positions_right: list[float] = []

    spacing = 1.0
    pos = 0.0
    for group, items_left in model_data.items():
        items_right = dlkgs_data[group]
        group_len = len(items_left)
        group_colors = color_palette[:group_len]

        for i, key in enumerate(items_left.keys()):
            labels.append(key)
            positions_left.append(pos + i - 0.2)
            values_left.append(items_left[key])
            bar_colors.append(group_colors[i])

            positions_right.append(pos + i + 0.2)
            values_right.append(items_right[key])

        pos += group_len + spacing

    unified_font_size = 18
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = unified_font_size

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(
        positions_left,
        values_left,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.4,
        label=model_name,
    )
    ax.bar(
        positions_right,
        values_right,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.4,
        hatch="///",
        alpha=0.8,
        label="DLKGs",
    )

    uniform_y = max(max(values_left), max(values_right)) + 0.04
    h = 0.015
    for i, label in enumerate(labels):
        x1 = positions_left[i]
        x2 = positions_right[i]
        star = p_to_star(p_values[label])
        ax.plot([x1, x1, x2, x2], [uniform_y, uniform_y + h, uniform_y + h, uniform_y], lw=1.2, color="black")
        offset = 0.002 if star == "ns" else 0.005
        ax.text(
            (x1 + x2) / 2,
            uniform_y + h + offset,
            star,
            ha="center",
            va="bottom",
            color="black",
            fontsize=unified_font_size,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.set_facecolor("white")
    ax.spines["left"].set_bounds(0.2, 1.0)

    ax.set_xticks([(left + right) / 2 for left, right in zip(positions_left, positions_right)])
    ax.set_xticklabels(labels, fontsize=unified_font_size, rotation=0, ha="center")
    ax.set_ylim(0.2, 1.2)
    ax.set_yticks(np.arange(0.2, 1.01, 0.1))
    ax.set_ylabel(f"Accuracy ( {model_name} vs. DLKGs )", fontsize=unified_font_size, labelpad=10)

    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=1.5, label=model_name),
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=1.5, hatch="///", label="DLKGs"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.13),
        ncol=1,
        fontsize=unified_font_size,
        frameon=False,
        handlelength=1.5,
        handleheight=1.0,
        handletextpad=0.5,
        borderaxespad=0.1,
        borderpad=0.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def p_to_external_star(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_external_gemini_vs_dlkgs(result_root: Path, output_path: Path) -> None:
    master = build_external_master(result_root)
    rows: list[dict[str, object]] = []
    for category in CATEGORIES:
        config = EXTERNAL_CATEGORY_CONFIG[category]
        for subgroup, display in zip(config["subgroups"], config["display"]):
            subset = master[(master["category"] == category) & (master["subgroup"] == subgroup)]
            if subset.empty:
                continue
            n00, n01, n10, n11, p_value = compute_mcnemar(subset["correct_Gemini"], subset["correct_DLKGS"])
            rows.append(
                {
                    "category": category,
                    "display": display,
                    "gemini_accuracy": float(subset["correct_Gemini"].mean()),
                    "dlkgs_accuracy": float(subset["correct_DLKGS"].mean()),
                    "p_value": p_value,
                }
            )

    color_palette = ["#4DBBD5", "#E64B35", "#00A087", "#3C5488"]
    labels: list[str] = []
    values_gemini: list[float] = []
    values_dlkgs: list[float] = []
    positions_gemini: list[float] = []
    positions_dlkgs: list[float] = []
    colors: list[str] = []
    p_values: list[float] = []

    pos = 0.0
    spacing = 1.0
    for category in CATEGORIES:
        category_rows = [row for row in rows if row["category"] == category]
        for index, row in enumerate(category_rows):
            labels.append(str(row["display"]))
            positions_gemini.append(pos + index - 0.2)
            positions_dlkgs.append(pos + index + 0.2)
            values_gemini.append(float(row["gemini_accuracy"]))
            values_dlkgs.append(float(row["dlkgs_accuracy"]))
            colors.append(color_palette[index % len(color_palette)])
            p_values.append(float(row["p_value"]))
        pos += len(category_rows) + spacing

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 16
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(positions_gemini, values_gemini, color=colors, edgecolor="black", linewidth=0.8, width=0.4, label="Gemini")
    ax.bar(
        positions_dlkgs,
        values_dlkgs,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.4,
        hatch="///",
        alpha=0.8,
        label="DLKGs",
    )

    global_max = max(values_gemini + values_dlkgs)
    uniform_y = min(1.08, global_max + 0.05)
    h = 0.015
    for index, p_value in enumerate(p_values):
        x1, x2 = positions_gemini[index], positions_dlkgs[index]
        ax.plot([x1, x1, x2, x2], [uniform_y, uniform_y + h, uniform_y + h, uniform_y], lw=1.2, color="black")
        ax.text(
            (x1 + x2) / 2,
            uniform_y + h + 0.004,
            p_to_external_star(p_value),
            ha="center",
            va="bottom",
            color="black",
            fontsize=16,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.set_facecolor("white")
    ax.set_xticks([(left + right) / 2 for left, right in zip(positions_gemini, positions_dlkgs)])
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.set_ylabel("Accuracy ( Gemini vs. DLKGs )", labelpad=10)
    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=1.5, label="Gemini"),
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=1.5, hatch="///", label="DLKGs"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 1.13), frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot subgroup accuracy comparisons between one MLLM and DLKGs.")
    parser.add_argument("--dataset", choices=["test", "external"], default="test")
    parser.add_argument(
        "--model",
        choices=["all", *MODEL_CONFIGS.keys()],
        default="all",
        help="Model to plot. Use 'all' to generate all four model comparisons.",
    )
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--p-value-path", type=Path, default=P_VALUE_PATH, help="McNemar p-value table for test-set subgroup plots.")
    parser.add_argument("--output-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Output file path for a single selected model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "external":
        if args.model not in {"all", "gemini"}:
            raise SystemExit("--dataset external only supports --model gemini.")
        output_path = args.output or args.output_dir / "Gemini" / "accuracy_comparison_Gemini_DLKGs.png"
        plot_external_gemini_vs_dlkgs(args.result_root or DEFAULT_EXTERNAL_RESULT_ROOT, output_path)
        print(f"saved: {output_path}")
        return

    if args.output is not None and args.model == "all":
        raise SystemExit("--output can only be used when --model is a single model.")

    result_root = args.result_root or RESULT_ROOT
    model_keys = list(MODEL_CONFIGS) if args.model == "all" else [args.model]
    for model_key in model_keys:
        output_path = (
            args.output
            if args.output is not None
            else args.output_dir / MODEL_CONFIGS[model_key]["folder"] / MODEL_CONFIGS[model_key]["output"]
        )
        plot_model_vs_dlkgs(model_key, output_path, result_root, args.p_value_path)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
