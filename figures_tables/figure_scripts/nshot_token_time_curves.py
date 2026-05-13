from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result" / "nshot_latest"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "image"

MODEL_ROOTS = {
    "Gemini3": "gemini-3-pro-preview_2026_1_8",
    "GPT5.2": "gpt-5.2-chat-latest_2026-1-13",
    "Claude4.5": "claude-opus-4-5-20251101-thinking_2026-1-13",
    "Grok4": "grok-4-0709_2026-1-13",
}

COLORS = {
    "Gemini3": "#4DBBD5",
    "GPT5.2": "#E64B35",
    "Claude4.5": "#00A087",
    "Grok4": "#3C5488",
}

NUM_GROUPS = 6
NUM_STAGES = 6
ROUNDS = np.array([1, 2, 3, 4, 5, 6])
X_LABELS = ["2-shot", "6-shot", "10-shot", "14-shot", "18-shot", "22-shot"]
TEST_IMG_NUM = [8, 8, 8, 8, 8, 8]
EXAMPLE_IMG_NUM = [16 + 32 * i for i in range(6)]


def load_stage_means(result_root: Path, metric: str) -> dict[str, list[float]]:
    data_dict: dict[str, list[float]] = {}

    for model, folder in MODEL_ROOTS.items():
        root = result_root / folder
        stage_values: list[float] = []
        for stage in range(NUM_STAGES):
            metric_values: list[float] = []
            for group in range(1, NUM_GROUPS + 1):
                file_path = root / f"cv_fixed{group}_stage{stage}.xlsx"
                df = pd.read_excel(file_path)
                if metric not in df.columns:
                    raise ValueError(f"{file_path} missing column: {metric}")
                metric_values.extend(df[metric].dropna().tolist())
            stage_values.append(float(np.mean(metric_values)))
        data_dict[model] = stage_values

    return data_dict


def plot_curve(
    data_dict: dict[str, list[float]],
    ylabel: str,
    output_path: Path,
    scale_k: bool = False,
    ylim: int | None = None,
) -> None:
    unified_font_size = 19
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = unified_font_size

    fig, ax1 = plt.subplots(figsize=(8, 5))

    for model, values in data_dict.items():
        y_values = np.array(values) / 1000.0 if scale_k else np.array(values)
        ax1.plot(
            ROUNDS,
            y_values,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=model,
            color=COLORS[model],
        )

    ax1.set_ylabel(ylabel, fontsize=unified_font_size)
    if ylim is not None:
        ax1.set_ylim(0, ylim)
        y_tick_step = 25 if scale_k else 5
        y_ticks = np.arange(0, ylim + 1, y_tick_step)
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(y_ticks, fontsize=unified_font_size)

    ax1.set_xticks(ROUNDS)
    ax1.set_xticklabels(X_LABELS, fontsize=unified_font_size)
    ax1.tick_params(axis="both", width=1.8)

    ax1.grid(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.8)
    ax1.spines["bottom"].set_linewidth(1.8)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Number of images per batch", fontsize=unified_font_size)
    ax2.plot(ROUNDS, TEST_IMG_NUM, linestyle="--", linewidth=2.0, color="#F39B7F", marker="s", markersize=6)
    ax2.plot(ROUNDS, EXAMPLE_IMG_NUM, linestyle="--", linewidth=2.0, color="#8491B4", marker="^", markersize=6)
    ax2.set_ylim(0, 200)
    ax2.set_yticks(np.arange(0, 201, 40))
    ax2.set_yticklabels(np.arange(0, 201, 40), fontsize=unified_font_size)
    ax2.tick_params(axis="y", width=1.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(1.8)

    legend_handles = [
        Line2D([], [], color=COLORS["Gemini3"], marker="o", linewidth=2.2),
        Line2D([], [], color=COLORS["GPT5.2"], marker="o", linewidth=2.2),
        Line2D([], [], color=COLORS["Claude4.5"], marker="o", linewidth=2.2),
        Line2D([], [], color=COLORS["Grok4"], marker="o", linewidth=2.2),
        Line2D([], [], color="#8491B4", marker="^", linestyle="--", linewidth=2.0),
        Line2D([], [], color="#F39B7F", marker="s", linestyle="--", linewidth=2.0),
        Line2D([], [], color="none"),
        Line2D([], [], color="none"),
    ]
    legend_labels = ["Gemini3", "GPT5.2", "Claude4.5", "Grok4", "Sample images", "Test images", "", ""]

    ax1.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        fontsize=unified_font_size - 2,
        frameon=False,
        ncol=2,
        columnspacing=1.5,
        labelspacing=0.6,
        borderpad=0.8,
        handletextpad=0.5,
        bbox_to_anchor=(0.0, 1.0),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot token and response-time curves for n-shot experiments.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = load_stage_means(args.result_root, "tokens")
    times = load_stage_means(args.result_root, "response_time")

    plot_curve(tokens, "Tokens per batch (k)", args.output_dir / "tokens_curve.png", scale_k=True, ylim=225)
    plot_curve(times, "Response time per batch (s)", args.output_dir / "response_time_curve.png", ylim=30)

    print(f"saved: {args.output_dir / 'tokens_curve.png'}")
    print(f"saved: {args.output_dir / 'response_time_curve.png'}")


if __name__ == "__main__":
    main()
