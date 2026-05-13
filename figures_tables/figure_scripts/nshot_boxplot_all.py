import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURES_TABLES_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "result"
SCORE_ROOT = PROJECT_ROOT / "dataset" / "nshotdataset" / "scoredata"
OUTPUT_DIR = FIGURES_TABLES_ROOT / "image"

NUM_GROUPS = 6
NUM_STAGES = 5
SHOT_LABELS = ["2-shot", "6-shot", "10-shot", "14-shot", "18-shot", "22-shot"]
ORDERED_MODELS = ["Gemini3", "GPT5.2", "Claude4.5", "Grok4"]
UNIFIED_FONT_SIZE = 14

PALETTE = {
    "Gemini3": "#4DBBD5",
    "GPT5.2": "#E64B35",
    "Claude4.5": "#00A087",
    "Grok4": "#3C5488",
}

RESULT_SETS = {
    "nshot": {
        "subdir": "nshot",
        "output_name": "nshot_boxplot.png",
        "legend_loc": "upper right",
        "model_dirs": {
            "Gemini3": "gemini-3",
            "GPT5.2": "gpt-5.2",
            "Claude4.5": "claude-4-5",
            "Grok4": "grok-4",
        },
    },
    "nshot_latest": {
        "subdir": "nshot_latest",
        "output_name": "nshot_latest_boxplot.png",
        "legend_loc": "lower right",
        "model_dirs": {
            "Gemini3": "gemini-3-pro-preview_2026_1_8",
            "GPT5.2": "gpt-5.2-chat-latest_2026-1-13",
            "Claude4.5": "claude-opus-4-5-20251101-thinking_2026-1-13",
            "Grok4": "grok-4-0709_2026-1-13",
        },
    },
}

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


def load_true_scores(score_root, fixed_group):
    true_json_path = score_root / f"data{fixed_group}" / "true_scores.json"
    if not true_json_path.exists():
        raise FileNotFoundError(f"True-score file not found: {true_json_path}")

    with open(true_json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def compute_accuracy_matrix_for_model(result_root, score_root):
    if not result_root.exists():
        raise FileNotFoundError(f"Model result directory not found: {result_root}")

    acc_matrix = np.zeros((NUM_GROUPS, NUM_STAGES + 1))

    for fixed_group in range(1, NUM_GROUPS + 1):
        true_scores = load_true_scores(score_root, fixed_group)

        for stage in range(0, NUM_STAGES + 1):
            file_path = result_root / f"cv_fixed{fixed_group}_stage{stage}.xlsx"
            if not file_path.exists():
                raise FileNotFoundError(f"Result file not found: {file_path}")

            df = pd.read_excel(file_path)
            correct = 0
            total = 0

            for _, row in df.iterrows():
                image_name = row["image"]
                if not isinstance(image_name, str) or "_" not in image_name:
                    continue

                file_name = image_name.split("_", 1)[1]
                truth = true_scores.get(file_name, None)
                if truth is not None:
                    total += 1
                    if row["score"] == truth:
                        correct += 1

            acc_matrix[fixed_group - 1, stage] = correct / total if total > 0 else 0

    return acc_matrix


def collect_model_matrices(config, result_root, score_root):
    result_set_root = result_root / config["subdir"]
    return {
        model_name: compute_accuracy_matrix_for_model(result_set_root / model_dir, score_root)
        for model_name, model_dir in config["model_dirs"].items()
    }


def plot_overall_boxplot(model_matrices, output_path, legend_loc):
    box_width = 0.14
    box_gap = 0.04
    plot_data = []
    positions = []
    box_colors = []

    for stage in range(NUM_STAGES + 1):
        for model_index, model in enumerate(ORDERED_MODELS):
            pos = stage + (model_index - 1.5) * (box_width + box_gap)
            positions.append(pos)
            plot_data.append(model_matrices[model][:, stage])
            box_colors.append(PALETTE[model])

    plt.rcParams["font.size"] = UNIFIED_FONT_SIZE
    plt.figure(figsize=(12, 4))
    ax = plt.gca()

    bplot = ax.boxplot(
        plot_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        boxprops={"linewidth": 1.8, "color": "black"},
        whiskerprops={"linewidth": 1.8, "color": "black"},
        capprops={"linewidth": 1.8, "color": "black"},
        medianprops={"linewidth": 1.8, "color": "black"},
        flierprops={"marker": "d", "markerfacecolor": "gray", "markersize": 5},
    )

    for patch, color in zip(bplot["boxes"], box_colors):
        patch.set_facecolor(color)

    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0.0, 1.01, 0.2), fontsize=UNIFIED_FONT_SIZE)
    plt.xticks(range(NUM_STAGES + 1), labels=SHOT_LABELS, fontsize=UNIFIED_FONT_SIZE)
    plt.xlim(-0.5, NUM_STAGES + 0.5)
    plt.ylabel("Accuracy ( 7-class )", fontsize=UNIFIED_FONT_SIZE)
    plt.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.tick_params(axis="both", width=1.6, labelsize=UNIFIED_FONT_SIZE)

    legend_handles = [
        mpatches.Patch(facecolor=PALETTE[model], edgecolor="black", linewidth=1.8, label=model)
        for model in ORDERED_MODELS
    ]
    ax.legend(
        handles=legend_handles,
        loc=legend_loc,
        fontsize=UNIFIED_FONT_SIZE,
        frameon=False,
        prop={"size": UNIFIED_FONT_SIZE},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"saved: {output_path}")
    plt.show()


def run_one(result_set, result_root, score_root, output_dir):
    config = RESULT_SETS[result_set]
    model_matrices = collect_model_matrices(config, result_root, score_root)
    plot_overall_boxplot(
        model_matrices,
        output_dir / config["output_name"],
        config["legend_loc"],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Draw n-shot 7-class overall boxplots.")
    parser.add_argument(
        "--result-set",
        choices=["nshot", "nshot_latest", "both"],
        default="both",
        help="Use nshot for nshot_boxplot, nshot_latest for nshot_latest_boxplot, or both.",
    )
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT, help="Directory containing nshot and nshot_latest folders.")
    parser.add_argument("--score-root", type=Path, default=SCORE_ROOT, help="Directory containing nshot scoredata JSON files.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    result_sets = ["nshot", "nshot_latest"] if args.result_set == "both" else [args.result_set]
    for result_set in result_sets:
        run_one(result_set, args.result_root, args.score_root, args.output_dir)


if __name__ == "__main__":
    main()
