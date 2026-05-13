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
FOUR_CLASS_JSON = SCORE_ROOT / "3c_scores.json"
OUTPUT_DIR = FIGURES_TABLES_ROOT / "image"

ROOTS = {
    "Gemini3": RESULT_ROOT / "gemini-3",
    "GPT5.2": RESULT_ROOT / "gpt-5.2",
    "Claude4.5": RESULT_ROOT / "claude-4-5",
    "Grok4": RESULT_ROOT / "grok-4",
}

NUM_GROUPS = 6
NUM_STAGES = 5
SHOT_LABELS = ["2-shot", "6-shot", "10-shot", "14-shot", "18-shot", "22-shot"]
UNIFIED_FONT_SIZE = 18

COLORS = {
    "Gemini3": "#4DBBD5",
    "GPT5.2": "#E64B35",
    "Claude4.5": "#00A087",
    "Grok4": "#3C5488",
}

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = UNIFIED_FONT_SIZE


def map_three_class(label):
    if label in ["NO", "C-0", "C-1", "0"]:
        return 0
    if label in ["C-2", "C-3", "1"]:
        return 1
    if label in ["O-1", "O-2", "O-3", "2", "3"]:
        return 2
    return None


def load_three_class_mapping(score_root):
    four_class_json = score_root / "3c_scores.json"
    if not four_class_json.exists():
        raise FileNotFoundError(f"Three-class mapping file not found: {four_class_json}")
    with open(four_class_json, "r", encoding="utf-8") as file:
        return json.load(file)


def load_true_scores(score_root, fixed_group):
    true_json_path = score_root / f"data{fixed_group}" / "true_scores.json"
    if not true_json_path.exists():
        raise FileNotFoundError(f"True-score file not found: {true_json_path}")
    with open(true_json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def compute_accuracy_matrix(result_root, score_root, threeclass=False):
    if not result_root.exists():
        raise FileNotFoundError(f"Result directory not found: {result_root}")

    acc_matrix = np.zeros((NUM_GROUPS, NUM_STAGES + 1))
    three_class_mapping = load_three_class_mapping(score_root) if threeclass else None

    for fixed_group in range(1, NUM_GROUPS + 1):
        true_scores = load_true_scores(score_root, fixed_group)

        for stage in range(0, NUM_STAGES + 1):
            file_path = result_root / f"cv_fixed{fixed_group}_stage{stage}.xlsx"
            if not file_path.exists():
                continue

            df = pd.read_excel(file_path)
            correct = 0
            total = 0

            for _, row in df.iterrows():
                image_name = row["image"]
                if not isinstance(image_name, str):
                    continue

                file_name = image_name.split("_", 1)[1]
                pred = row["score"]
                truth = true_scores.get(file_name, None)

                if threeclass:
                    pred = map_three_class(pred)
                    truth = map_three_class(three_class_mapping.get(file_name, None))

                if truth is not None and pred is not None:
                    total += 1
                    if pred == truth:
                        correct += 1

            acc_matrix[fixed_group - 1, stage] = correct / total if total > 0 else 0

    return pd.DataFrame(
        acc_matrix,
        index=[f"Fold{i}" for i in range(1, NUM_GROUPS + 1)],
        columns=[f"Stage{stage}" for stage in range(NUM_STAGES + 1)],
    )


def plot_avg_curves(acc_dict, y_label, output_path, legend_loc, legend_anchor=None):
    rounds = range(NUM_STAGES + 1)
    plt.figure(figsize=(8, 5))

    for model, acc_matrix in acc_dict.items():
        avg_acc = acc_matrix.mean(axis=0)
        plt.plot(
            rounds,
            avg_acc,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=f"{model} ( Average )",
            color=COLORS[model],
        )

    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0.0, 1.01, 0.2), fontsize=UNIFIED_FONT_SIZE)
    plt.xticks(range(NUM_STAGES + 1), labels=SHOT_LABELS, fontsize=UNIFIED_FONT_SIZE)
    plt.ylabel(y_label, fontsize=UNIFIED_FONT_SIZE)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.tick_params(axis="both", width=1.6, labelsize=UNIFIED_FONT_SIZE)

    legend_kwargs = {
        "loc": legend_loc,
        "fontsize": UNIFIED_FONT_SIZE,
        "frameon": False,
        "prop": {"size": UNIFIED_FONT_SIZE},
    }
    if legend_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = legend_anchor
    plt.legend(**legend_kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"saved: {output_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Draw average n-shot accuracy curves for all models.")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT, help="Directory containing nshot model folders.")
    parser.add_argument("--score-root", type=Path, default=SCORE_ROOT, help="Directory containing nshot scoredata JSON files.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    roots = {
        "Gemini3": args.result_root / "gemini-3",
        "GPT5.2": args.result_root / "gpt-5.2",
        "Claude4.5": args.result_root / "claude-4-5",
        "Grok4": args.result_root / "grok-4",
    }
    acc_dict_7 = {model: compute_accuracy_matrix(path, args.score_root, threeclass=False) for model, path in roots.items()}
    plot_avg_curves(
        acc_dict_7,
        "Accuracy ( 7-class )",
        args.output_dir / "cv_accuracy_curve_7class.png",
        legend_loc="upper right",
        legend_anchor=(1.0, 1.05),
    )

    acc_dict_3 = {model: compute_accuracy_matrix(path, args.score_root, threeclass=True) for model, path in roots.items()}
    plot_avg_curves(
        acc_dict_3,
        "Accuracy ( 3-class )",
        args.output_dir / "cv_accuracy_curve_3class.png",
        legend_loc="lower right",
    )


if __name__ == "__main__":
    main()
