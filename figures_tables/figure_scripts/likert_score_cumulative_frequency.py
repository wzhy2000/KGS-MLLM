from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


TABLE_DIR = Path(__file__).resolve().parents[1] / "table"
IMAGE_DIR = Path(__file__).resolve().parents[1] / "image"

DEFAULT_SCORE0 = TABLE_DIR / "score0.xlsx"
DEFAULT_SCORE1 = TABLE_DIR / "score1.xlsx"
DEFAULT_OUTPUT = IMAGE_DIR / "cumulative_frequency_plot.png"

MODELS = ["Gemini3", "GPT5.2", "Claude4.5", "Grok4"]
COLORS = {
    "Gemini3": "#4DBBD5",
    "GPT5.2": "#E64B35",
    "Claude4.5": "#00A087",
    "Grok4": "#3C5488",
}
RENAME_COLUMNS = {"Gemini": "Gemini3", "GPT": "GPT5.2", "Claude": "Claude4.5", "Grok": "Grok4"}


def load_scores(score0_path: Path, score1_path: Path) -> pd.DataFrame:
    df0 = pd.read_excel(score0_path)
    df1 = pd.read_excel(score1_path)
    df = pd.concat([df0, df1], ignore_index=True)
    return df[["Gemini", "GPT", "Claude", "Grok"]].rename(columns=RENAME_COLUMNS)


def plot_cumulative_frequency(score0_path: Path, score1_path: Path, output_path: Path) -> None:
    df = load_scores(score0_path, score1_path)

    unified_font_size = 20
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = unified_font_size
    rcParams["font.size"] = unified_font_size
    rcParams["axes.labelsize"] = unified_font_size
    rcParams["xtick.labelsize"] = unified_font_size
    rcParams["ytick.labelsize"] = unified_font_size
    rcParams["legend.fontsize"] = unified_font_size
    rcParams["axes.titlesize"] = unified_font_size

    fig, ax = plt.subplots(figsize=(10, 7))

    for model in MODELS:
        scores = df[model].dropna().sort_values()
        cum_freq = np.arange(1, len(scores) + 1) / len(scores) * 100
        if scores.iloc[-1] == 5:
            scores = np.append(scores, 5.05)
            cum_freq = np.append(cum_freq, cum_freq[-1])
        ax.plot(scores, cum_freq, label=model, color=COLORS[model], linewidth=2)

    ax.set_xlabel("Likert score", fontsize=unified_font_size)
    ax.set_ylabel("Cumulative frequency (%)", fontsize=unified_font_size)
    ax.set_xlim(0.95, 5.05)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 110, 20))
    ax.legend(loc="upper left", fontsize=unified_font_size, frameon=False)
    ax.grid(False)
    ax.set_facecolor("white")
    fig.set_facecolor("white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cumulative frequency curves for Likert scores.")
    parser.add_argument("--score0", type=Path, default=DEFAULT_SCORE0)
    parser.add_argument("--score1", type=Path, default=DEFAULT_SCORE1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_cumulative_frequency(args.score0, args.score1, args.output)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
