from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORE_ROOT = PROJECT_ROOT / "dataset" / "nshotdataset" / "scoredata"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "image"

RESULT_SETS = {
    "nshot": {
        "root": PROJECT_ROOT / "result" / "nshot",
        "prefix": "nshot",
        "models": {
            "gemini": {"folder": "gemini-3", "display": "Gemini3", "image_folder": "Gemini"},
            "gpt": {"folder": "gpt-5.2", "display": "GPT5.2", "image_folder": "GPT"},
            "claude": {"folder": "claude-4-5", "display": "Claude4.5", "image_folder": "Claude"},
            "grok": {"folder": "grok-4", "display": "Grok4", "image_folder": "Grok"},
        },
    },
    "nshot_latest": {
        "root": PROJECT_ROOT / "result" / "nshot_latest",
        "prefix": "nshot_latest",
        "models": {
            "gemini": {"folder": "gemini-3-pro-preview_2026_1_8", "display": "Gemini3", "image_folder": "Gemini"},
            "gpt": {"folder": "gpt-5.2-chat-latest_2026-1-13", "display": "GPT5.2", "image_folder": "GPT"},
            "claude": {
                "folder": "claude-opus-4-5-20251101-thinking_2026-1-13",
                "display": "Claude4.5",
                "image_folder": "Claude",
            },
            "grok": {"folder": "grok-4-0709_2026-1-13", "display": "Grok4", "image_folder": "Grok"},
        },
    },
}

NUM_GROUPS = 6
NUM_STAGES = 5
X_LABELS = ["2-shot", "6-shot", "10-shot", "14-shot", "18-shot", "22-shot"]


def extract_filename(image_value: object) -> str | None:
    if not isinstance(image_value, str):
        return None
    if "_" in image_value:
        return image_value.split("_", 1)[1]
    return Path(image_value).name


def map_3class(label: object) -> int | None:
    if label in ["C-0", "C-1", "0"]:
        return 0
    if label in ["C-2", "C-3", "1"]:
        return 1
    if label in ["O-1", "O-2", "O-3", "2", "3"]:
        return 2
    return None


def compute_accuracy_matrix(result_root: Path, score_root: Path, class_mode: str) -> pd.DataFrame:
    acc_matrix = np.zeros((NUM_GROUPS, NUM_STAGES + 1))
    three_class_path = score_root / "3c_scores.json"
    three_class_truth = json.loads(three_class_path.read_text(encoding="utf-8")) if class_mode == "3" else None

    for fixed_group in range(1, NUM_GROUPS + 1):
        true_scores_path = score_root / f"data{fixed_group}" / "true_scores.json"
        true_scores = json.loads(true_scores_path.read_text(encoding="utf-8"))

        for stage in range(NUM_STAGES + 1):
            file_path = result_root / f"cv_fixed{fixed_group}_stage{stage}.xlsx"
            if not file_path.exists():
                continue

            df = pd.read_excel(file_path)
            correct = 0
            total = 0

            for _, row in df.iterrows():
                filename = extract_filename(row["image"])
                if filename is None:
                    continue

                if class_mode == "7":
                    pred = row["score"]
                    truth = true_scores.get(filename)
                    if truth is None:
                        continue
                else:
                    assert three_class_truth is not None
                    pred = map_3class(row["score"])
                    truth = map_3class(three_class_truth.get(filename))
                    if truth is None or pred is None:
                        continue

                total += 1
                if pred == truth:
                    correct += 1

            acc_matrix[fixed_group - 1, stage] = correct / total if total > 0 else 0

    index_prefix = "Exp." if class_mode == "7" else "Fold"
    return pd.DataFrame(
        acc_matrix,
        index=[f"{index_prefix}{i}" for i in range(1, NUM_GROUPS + 1)],
        columns=[f"Stage{j}" for j in range(NUM_STAGES + 1)],
    )


def add_split_legend(
    ax: plt.Axes,
    lines: dict[str, object],
    class_mode: str,
    font_size: int,
    force_lower_right: bool = False,
) -> None:
    handles_left = [lines["Exp.1"], lines["Exp.2"], lines["Exp.3"], lines["Exp.4"]]
    labels_left = ["Exp.1", "Exp.2", "Exp.3", "Exp.4"]
    handles_right = [lines["Exp.5"], lines["Exp.6"], lines["Average"]]
    labels_right = ["Exp.5", "Exp.6", "Average"]

    if class_mode == "7" and not force_lower_right:
        loc = "upper right"
        y_anchor = 0.98
    else:
        loc = "lower right"
        y_anchor = 0.02

    leg_right = ax.legend(
        handles_right,
        labels_right,
        loc=loc,
        fontsize=font_size,
        frameon=False,
        bbox_to_anchor=(1.00, y_anchor),
        borderaxespad=0.8,
    )
    ax.add_artist(leg_right)
    ax.legend(
        handles_left,
        labels_left,
        loc=loc,
        fontsize=font_size,
        frameon=False,
        bbox_to_anchor=(0.78, y_anchor),
        borderaxespad=0.0,
    )


def plot_accuracy_curve(
    acc_matrix: pd.DataFrame,
    model_display: str,
    class_mode: str,
    output_path: Path,
    force_lower_right_legend: bool = False,
) -> None:
    font_size = 18
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = font_size

    rounds = range(NUM_STAGES + 1)
    fig, ax = plt.subplots(figsize=(11, 6.0))

    lines: dict[str, object] = {}
    for i in range(NUM_GROUPS):
        line, = ax.plot(
            rounds,
            acc_matrix.iloc[i, :],
            marker="o",
            linestyle="--",
            linewidth=2.0,
            markersize=6,
            label=f"Exp.{i + 1}",
        )
        lines[f"Exp.{i + 1}"] = line

    avg_acc = acc_matrix.mean(axis=0)
    avg_line, = ax.plot(
        rounds,
        avg_acc,
        marker="s",
        linestyle="-",
        color="red",
        linewidth=2.4,
        markersize=7,
        label="Average",
    )
    lines["Average"] = avg_line

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xticks(list(rounds))
    ax.set_xticklabels(X_LABELS, fontsize=font_size)
    ax.set_ylabel(f"Accuracy ( {model_display} )", fontsize=font_size)
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.tick_params(axis="both", width=1.6, labelsize=font_size)

    add_split_legend(ax, lines, class_mode, font_size, force_lower_right=force_lower_right_legend)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def iter_model_keys(model_arg: str) -> list[str]:
    if model_arg == "all":
        return ["gemini", "gpt", "claude", "grok"]
    return [model_arg]


def iter_class_modes(class_mode_arg: str) -> list[str]:
    if class_mode_arg == "both":
        return ["7", "3"]
    return [class_mode_arg]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate n-shot accuracy line plots for 7-class Kimura-Takemoto "
            "and KGS-A 3-class tasks."
        )
    )
    parser.add_argument("--result-set", choices=RESULT_SETS.keys(), default="nshot")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT, help="Directory containing nshot and nshot_latest folders.")
    parser.add_argument("--model", choices=["all", "gemini", "gpt", "claude", "grok"], default="all")
    parser.add_argument("--class-mode", choices=["both", "7", "3"], default="both")
    parser.add_argument("--score-root", type=Path, default=DEFAULT_SCORE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--group-by-result-set",
        action="store_true",
        help="Save under output-root/result-set/class-folder/model.png instead of model subfolders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_config = RESULT_SETS[args.result_set]
    result_root = args.result_root / args.result_set

    for model_key in iter_model_keys(args.model):
        model_config = result_config["models"][model_key]
        model_result_root = result_root / model_config["folder"]
        for class_mode in iter_class_modes(args.class_mode):
            acc_matrix = compute_accuracy_matrix(model_result_root, args.score_root, class_mode)
            if args.group_by_result_set:
                class_folder = "6fold_7class" if class_mode == "7" else "6fold_3class"
                file_name = f"{model_config['image_folder']}.png"
                output_path = args.output_root / args.result_set / class_folder / file_name
            else:
                class_label = "7class" if class_mode == "7" else "3class"
                file_name = f"{result_config['prefix']}_accuracy_{class_label}.png"
                output_path = args.output_root / model_config["image_folder"] / file_name
            force_lower_right_legend = args.result_set == "nshot_latest" and class_mode == "7"
            plot_accuracy_curve(
                acc_matrix,
                model_config["display"],
                class_mode,
                output_path,
                force_lower_right_legend=force_lower_right_legend,
            )
            print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
