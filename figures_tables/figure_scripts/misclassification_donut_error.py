from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS_FILE = PROJECT_ROOT / "result" / "testdataset_doctorscore" / "misclassification_statistics.xlsx"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "image"

ERROR_LABELS = ["KD", "IKC", "IDR", "MKC", "RA"]
ERROR_ROW_KEYWORDS = [
    "Knowledge Deficiency",
    "Ignoring Key Clues",
    "Inadequate Diagnostic Reasoning",
    "Misinterpretation of Key Clues",
    "Refusing to Answer or Others",
]
MODEL_NAME_MAP = {
    "Gemini": "Gemini3",
    "GPT": "GPT5.2",
    "Claude": "Claude4.5",
    "Grok": "Grok4",
}
MODEL_FOLDERS = {
    "Gemini3": "Gemini",
    "GPT5.2": "GPT",
    "Claude4.5": "Claude",
    "Grok4": "Grok",
}
COLORS = ["#A6CEE3", "#1F78B4", "#B2DF8A", "#FB9A99", "#999999"]


def load_misclassification_counts(stats_file: Path) -> dict[str, list[int]]:
    raw = pd.read_excel(stats_file, sheet_name=0, header=None)
    models: dict[str, list[int]] = {}

    for row_idx, row in raw.iterrows():
        if str(row.iloc[0]).strip() != "错误统计":
            continue

        model_key = str(row.iloc[1]).strip()
        model_name = MODEL_NAME_MAP.get(model_key)
        if model_name is None:
            continue

        values: list[int] = []
        for offset, keyword in enumerate(ERROR_ROW_KEYWORDS, start=1):
            error_row = raw.iloc[row_idx + offset]
            label_text = str(error_row.iloc[0])
            if keyword not in label_text:
                raise ValueError(f"Unexpected error row at {stats_file}: {label_text!r}")
            values.append(int(error_row.iloc[7]))
        models[model_name] = values

    missing = [model for model in MODEL_NAME_MAP.values() if model not in models]
    if missing:
        raise ValueError(f"Missing model block(s) in {stats_file}: {missing}")

    return models


def plot_donut(model_name: str, values: list[int], output_path: Path) -> None:
    unified_font_size = 20
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = unified_font_size

    total = sum(values)
    fig, ax = plt.subplots(figsize=(5, 5))

    wedges, _ = ax.pie(
        values,
        radius=1,
        labels=None,
        colors=COLORS,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.5, "edgecolor": "white"},
    )

    ax.pie(
        [1],
        radius=0.5,
        colors=["#f0f0f0"],
        labels=None,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.5, "edgecolor": "white"},
    )

    ax.text(0, 0, f"Total\n{total}", ha="center", va="center", fontsize=unified_font_size, color="black")

    for i, (wedge, value) in enumerate(zip(wedges, values)):
        theta = (wedge.theta1 + wedge.theta2) / 2
        x = 0.75 * np.cos(np.deg2rad(theta))
        y = 0.75 * np.sin(np.deg2rad(theta))
        pct = value / total * 100
        label = f"{ERROR_LABELS[i]}\n({value})\n{pct:.1f}%"
        ax.text(x, y, label, ha="center", va="center", fontsize=unified_font_size, color="black")

    ax.set_title(model_name, fontsize=unified_font_size)
    ax.axis("equal")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot donut charts for model misclassification error types.")
    parser.add_argument("--stats-file", type=Path, default=DEFAULT_STATS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Save images directly under output-dir instead of model subfolders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = load_misclassification_counts(args.stats_file)

    for model_name, values in counts.items():
        if args.flat_output:
            output_path = args.output_dir / f"{model_name}_donut_error.png"
        else:
            output_path = args.output_dir / MODEL_FOLDERS[model_name] / f"{model_name}_donut_error.png"
        plot_donut(model_name, values, output_path)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
