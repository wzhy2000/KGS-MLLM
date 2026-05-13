from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = Path(__file__).resolve().parents[1] / "table"
IMAGE_DIR = Path(__file__).resolve().parents[1] / "image"

MODEL_REPORT = TABLE_DIR / "model_group_average_report.txt"
DOCTOR_REPORT = TABLE_DIR / "doctor_group_average_report.txt"
DEFAULT_OUTPUT = IMAGE_DIR / "overall_classification_performance.png"

TITLE = "Overall Classification Performance of MLLMs, DLKGs, and Endoscopists"
CATEGORY_ORDER = ["A-all", "DR-all", "H-all", "IM-all", "N-all"]
CATEGORY_WEIGHTS = [600, 600, 450, 600, 450]

MODEL_BLOCKS = {
    "Gemini3": "result_gemini",
    "GPT5.2": "result_gpt",
    "Claude4.5": "result_claude",
    "Grok4": "result_grok",
    "DLKGs": "result_DLKGS",
}

DOCTOR_BLOCKS = {
    "Senior": "group_1_doctor1_5",
    "Junior": "group_2_doctor6_10",
    "Mixed": "groups_3",
}

MODELS = ["Gemini3", "GPT5.2", "Claude4.5", "Grok4", "DLKGs", "Senior", "Junior", "Mixed"]
METRICS = ["Accuracy (%)", "Specificity (%)", "Precision (%)", "Recall (%)", "F1 score (%)"]
METRIC_LABELS = ["Accuracy", "Specificity", "Precision", "Recall", "F1 score"]

COLORS = {
    "Gemini3": "#4DBBD5",
    "GPT5.2": "#E64B35",
    "Claude4.5": "#00A087",
    "Grok4": "#3C5488",
    "DLKGs": "#F39B7F",
    "Senior": "#8491B4",
    "Junior": "#91D1C2",
    "Mixed": "#DC0000",
}


def parse_report(report_path: Path, block_mapping: dict[str, str]) -> dict[str, dict[str, list[float]]]:
    text = report_path.read_text(encoding="utf-8")
    blocks: dict[str, dict[str, dict[str, float | int]]] = {}
    current_block: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            current_block = line[1:-1]
            blocks[current_block] = {}
            continue

        if current_block is None:
            continue

        if line.startswith("Sub-category") or line.startswith("="):
            continue

        if line.startswith(tuple(CATEGORY_ORDER)):
            parts = re.split(r"\s+", line)
            name = parts[0]
            blocks[current_block][name] = {
                "samples": int(parts[1]),
                "Accuracy (%)": float(parts[2].rstrip("%")) / 100,
                "Specificity (%)": float(parts[3].rstrip("%")) / 100,
                "Precision (%)": float(parts[4].rstrip("%")) / 100,
                "Recall (%)": float(parts[5].rstrip("%")) / 100,
                "F1 score (%)": float(parts[6].rstrip("%")) / 100,
            }

    parsed: dict[str, dict[str, list[float]]] = {}
    for output_name, block_name in block_mapping.items():
        if block_name not in blocks:
            raise KeyError(f"Block [{block_name}] not found in {report_path}")

        block = blocks[block_name]
        parsed[output_name] = {
            metric: [float(block[category][metric]) for category in CATEGORY_ORDER]
            for metric in METRICS
        }

    return parsed


def build_summary_table(model_report: Path = MODEL_REPORT, doctor_report: Path = DOCTOR_REPORT) -> pd.DataFrame:
    raw_data: dict[str, dict[str, list[float]]] = {}
    raw_data.update(parse_report(model_report, MODEL_BLOCKS))
    raw_data.update(parse_report(doctor_report, DOCTOR_BLOCKS))

    rows: list[dict[str, float | str]] = []
    for metric in METRICS:
        for model in MODELS:
            values = np.array(raw_data[model][metric], dtype=float)
            weighted_mean = np.average(values, weights=CATEGORY_WEIGHTS)
            weighted_variance = np.average((values - weighted_mean) ** 2, weights=CATEGORY_WEIGHTS)
            sem = np.sqrt(weighted_variance) / np.sqrt(len(values))
            rows.append({"Metric": metric, "Model": model, "Value": weighted_mean, "SEM": sem})

    return pd.DataFrame(rows)


def plot_overall_performance(df: pd.DataFrame, output_path: Path, show_title: bool = False) -> None:
    unified_font_size = 14
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = unified_font_size

    fig, ax = plt.subplots(figsize=(12, 3.6))
    x = np.arange(len(METRICS))
    width = 0.10

    for i, model in enumerate(MODELS):
        model_data = df[df["Model"] == model]
        ax.bar(
            x + i * width,
            model_data["Value"].to_numpy(),
            width,
            label=model,
            color=COLORS[model],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
            yerr=model_data["SEM"].to_numpy(),
            error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
        )

    ax.set_ylim(0.5, 1.0)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_xticks(x + width * 3.5)
    ax.set_xticklabels(METRIC_LABELS, rotation=0, ha="center")
    ax.set_ylabel("Score")

    if show_title:
        ax.set_title(TITLE, pad=12)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="both", width=1.5, length=5, labelsize=unified_font_size)

    ax.legend(
        loc="upper right",
        ncol=4,
        fontsize=unified_font_size,
        frameon=False,
        prop={"size": unified_font_size},
        bbox_to_anchor=(1.0, 1.04),
        handletextpad=0.5,
        columnspacing=0.5,
        labelspacing=0.3,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--model-report", type=Path, default=MODEL_REPORT)
    parser.add_argument("--doctor-report", type=Path, default=DOCTOR_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show-title", action="store_true", help="Add the figure title inside the saved image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_summary_table(args.model_report, args.doctor_report)
    plot_overall_performance(df, args.output, show_title=args.show_title)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
