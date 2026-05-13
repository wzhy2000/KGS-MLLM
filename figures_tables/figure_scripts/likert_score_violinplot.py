from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TABLE_DIR = Path(__file__).resolve().parents[1] / "table"
IMAGE_DIR = Path(__file__).resolve().parents[1] / "image"

DEFAULT_SCORE0 = TABLE_DIR / "score0.xlsx"
DEFAULT_SCORE1 = TABLE_DIR / "score1.xlsx"
DEFAULT_OUTPUT = IMAGE_DIR / "violinplot.png"

MODELS = ["Gemini3", "GPT5.2", "Claude4.5", "Grok4"]
COLORS = {
    "Gemini3": "#4DBBD5",
    "GPT5.2": "#E64B35",
    "Claude4.5": "#00A087",
    "Grok4": "#3C5488",
}
RENAME_COLUMNS = {"Gemini": "Gemini3", "GPT": "GPT5.2", "Claude": "Claude4.5", "Grok": "Grok4"}


def load_scores(score0_path: Path, score1_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df0 = pd.read_excel(score0_path).rename(columns=RENAME_COLUMNS)
    df1 = pd.read_excel(score1_path).rename(columns=RENAME_COLUMNS)
    df0["Group"] = "Wrong"
    df1["Group"] = "Correct"
    return df0, df1, pd.concat([df0, df1], ignore_index=True)


def plot_violin(score0_path: Path, score1_path: Path, output_path: Path, random_seed: int | None = None) -> None:
    if random_seed is not None:
        np.random.seed(random_seed)

    df0, df1, df = load_scores(score0_path, score1_path)

    unified_font_size = 20
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = unified_font_size

    offset = {"Correct": -0.2, "Wrong": 0.2}
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, model in enumerate(MODELS):
        for group in ["Correct", "Wrong"]:
            data = df[df["Group"] == group][model].dropna()
            x_pos = i + offset[group]
            violin = ax.violinplot(
                dataset=[data],
                positions=[x_pos],
                widths=0.35,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(COLORS[model])
                body.set_edgecolor(COLORS[model])
                body.set_alpha(0.8)

    for i, model in enumerate(MODELS):
        for group in ["Correct", "Wrong"]:
            data = df[df["Group"] == group][model].dropna()
            x_pos = i + offset[group]
            ax.scatter(np.random.normal(x_pos, 0.03, size=len(data)), data, color="black", s=10, alpha=0.5)

    for i, model in enumerate(MODELS):
        for group in ["Correct", "Wrong"]:
            data = df[df["Group"] == group][model].dropna()
            x_pos = i + offset[group]
            ax.boxplot(
                data,
                positions=[x_pos],
                widths=0.15,
                showcaps=False,
                boxprops={"color": "none"},
                whiskerprops={"color": "none"},
                medianprops={"color": "none"},
                showfliers=False,
            )

    for i, model in enumerate(MODELS):
        ax.scatter(i + offset["Correct"], df1[model].mean(), color="red", s=80, zorder=10)
        ax.scatter(i + offset["Wrong"], df0[model].mean(), color="red", s=80, zorder=10)

    y_line_pos = 5.2
    y_text_pos = 5.3
    tick_height = 0.15
    for i, model in enumerate(MODELS):
        x_left = i + offset["Correct"]
        x_right = i + offset["Wrong"]
        ax.plot([x_left, x_right], [y_line_pos, y_line_pos], color="black", lw=1.5)
        ax.plot([x_left, x_left], [y_line_pos, y_line_pos - tick_height], color="black", lw=1.5)
        ax.plot([x_right, x_right], [y_line_pos, y_line_pos - tick_height], color="black", lw=1.5)
        ax.text(i, y_text_pos, model, ha="center", va="bottom", fontsize=unified_font_size)

    new_xticks = []
    new_xticklabels = []
    for i in range(len(MODELS)):
        new_xticks.append(i + offset["Correct"])
        new_xticklabels.append("Correct   ")
        new_xticks.append(i + offset["Wrong"])
        new_xticklabels.append("  Wrong")

    ax.set_xticks(new_xticks)
    ax.set_xticklabels(new_xticklabels, fontsize=unified_font_size - 1)
    ax.spines["left"].set_bounds(0, 5)
    ax.set_ylabel("Likert score", fontsize=unified_font_size)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylim(0, 5.8)

    sns.despine()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Likert score violin distributions for correct and wrong cases.")
    parser.add_argument("--score0", type=Path, default=DEFAULT_SCORE0)
    parser.add_argument("--score1", type=Path, default=DEFAULT_SCORE1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_violin(args.score0, args.score1, args.output, random_seed=args.random_seed)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
