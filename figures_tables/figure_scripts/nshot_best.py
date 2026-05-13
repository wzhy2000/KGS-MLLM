import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURES_TABLES_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "result" / "nshot"
SCORE_ROOT = PROJECT_ROOT / "dataset" / "nshotdataset" / "scoredata"
OUTPUT_PATH = FIGURES_TABLES_ROOT / "image" / "nshot_best_7class.png"

MODEL_DIRS = {
    "Gemini3": RESULT_ROOT / "gemini-3",
    "GPT5.2": RESULT_ROOT / "gpt-5.2",
    "Claude4.5": RESULT_ROOT / "claude-4-5",
    "Grok4": RESULT_ROOT / "grok-4",
}

COLORS = {
    "Gemini3": "#4DBBD5",
    "GPT5.2": "#E64B35",
    "Claude4.5": "#00A087",
    "Grok4": "#3C5488",
}

FIXED_GROUP = 2
NUM_STAGES = 5
SHOT_LABELS = ["2-shot", "6-shot", "10-shot", "14-shot", "18-shot", "22-shot"]
UNIFIED_FONT_SIZE = 18

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = UNIFIED_FONT_SIZE


def load_true_scores(score_root):
    true_score_path = score_root / f"data{FIXED_GROUP}" / "true_scores.json"
    if not true_score_path.exists():
        raise FileNotFoundError(f"True-score file not found: {true_score_path}")

    with open(true_score_path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_stage_accuracy(result_file, true_scores):
    df = pd.read_excel(result_file)
    correct = 0
    total = 0

    for _, row in df.iterrows():
        image_name = row["image"]
        if not isinstance(image_name, str) or "_" not in image_name:
            continue

        file_name = image_name.split("_", 1)[1]
        truth = true_scores.get(file_name)
        if truth is None:
            continue

        total += 1
        if row["score"] == truth:
            correct += 1

    return correct / total if total else 0


def collect_best_group_accuracy(model_dirs, score_root):
    true_scores = load_true_scores(score_root)
    accuracy_by_model = {}

    for model, model_dir in MODEL_DIRS.items():
        if not model_dir.exists():
            raise FileNotFoundError(f"Model result directory not found: {model_dir}")

        stage_values = []
        for stage in range(NUM_STAGES + 1):
            result_file = model_dir / f"cv_fixed{FIXED_GROUP}_stage{stage}.xlsx"
            if not result_file.exists():
                raise FileNotFoundError(f"Result file not found: {result_file}")
            stage_values.append(calculate_stage_accuracy(result_file, true_scores))

        accuracy_by_model[model] = stage_values

    return accuracy_by_model


def plot_accuracy_curves(accuracy_by_model, output_path: Path):
    rounds = np.arange(NUM_STAGES + 1)
    plt.figure(figsize=(8, 5))

    for model, values in accuracy_by_model.items():
        plt.plot(
            rounds,
            values,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=model,
            color=COLORS[model],
        )

    plt.xticks(rounds, labels=SHOT_LABELS, fontsize=UNIFIED_FONT_SIZE)
    plt.ylabel("Accuracy ( 7-class )", fontsize=UNIFIED_FONT_SIZE)
    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0.0, 1.01, 0.2), fontsize=UNIFIED_FONT_SIZE)
    plt.xlim(-0.5, 5.5)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.tick_params(axis="both", width=1.6, labelsize=UNIFIED_FONT_SIZE)

    plt.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.05),
        fontsize=UNIFIED_FONT_SIZE,
        frameon=False,
        prop={"size": UNIFIED_FONT_SIZE},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"saved: {output_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Draw the best-fold n-shot 7-class accuracy curve.")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT, help="Directory containing nshot model folders.")
    parser.add_argument("--score-root", type=Path, default=SCORE_ROOT, help="Directory containing nshot scoredata JSON files.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dirs = {
        "Gemini3": args.result_root / "gemini-3",
        "GPT5.2": args.result_root / "gpt-5.2",
        "Claude4.5": args.result_root / "claude-4-5",
        "Grok4": args.result_root / "grok-4",
    }
    accuracy_by_model = collect_best_group_accuracy(model_dirs, args.score_root)
    for model, values in accuracy_by_model.items():
        print(f"{model}: {[round(value, 4) for value in values]}")
    plot_accuracy_curves(accuracy_by_model, args.output)


if __name__ == "__main__":
    main()
