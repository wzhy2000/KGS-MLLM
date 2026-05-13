import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


UNIFIED_FONT_SIZE = 18
FIGURES_TABLES_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "result" / "ICLperturbresult"
OUTPUT_PATH = FIGURES_TABLES_ROOT / "image" / "ICLtestdraw.png"

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = UNIFIED_FONT_SIZE

def build_experiments(result_root: Path):
    return {
        "D.Prompt": {
            "path": result_root / "prompt_all",
            "prefix": "result_prompt",
        },
        "S.Prompt": {
            "path": result_root / "prompt_e",
            "prefix": "result_prompt",
        },
        "Rev.Cat.": {
            "path": result_root / "prompt_R",
            "prefix": "result_prompt",
        },
        "Rev.Exp.": {
            "path": result_root / "train_R",
            "prefix": "result_train",
        },
        "Rotated": {
            "path": result_root / "imgtrain_R",
            "prefix": "result_img",
        },
        "Aug.": {
            "path": result_root / "img_transform",
            "prefix": "result_img",
        },
    }


def load_and_calculate_accuracy(result_root: Path):
    ground_truth_path = result_root / "data_test.xlsx"
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    gt_df = pd.read_excel(ground_truth_path)
    gt_df.columns = [c.strip() for c in gt_df.columns]
    gt_dict = dict(zip(gt_df["image"], gt_df["score"]))

    results = {"Model": [], "Mean_Acc": [], "Error": [], "Acc_List": []}

    for label, config in build_experiments(result_root).items():
        accuracies = []
        base_path = config["path"]
        prefix = config["prefix"]

        print(f"Processing {label}...")

        for index in range(1, 6):
            file_path = base_path / f"{prefix}_{index}.xlsx"
            if not file_path.exists():
                print(f"  Warning: File not found {file_path}, skipping.")
                continue

            try:
                df = pd.read_excel(file_path)
                df.columns = [c.strip() for c in df.columns]

                correct_count = 0
                total_count = 0
                for _, row in df.iterrows():
                    image_name = row["image"]
                    pred_score = row["score"]

                    if image_name in gt_dict:
                        total_count += 1
                        if str(pred_score).strip() == str(gt_dict[image_name]).strip():
                            correct_count += 1

                if total_count > 0:
                    accuracies.append(correct_count / total_count)

            except Exception as exc:
                print(f"  Error reading {file_path.name}: {exc}")

        if accuracies:
            results["Model"].append(label)
            results["Mean_Acc"].append(np.mean(accuracies))
            results["Error"].append(np.std(accuracies, ddof=1))
            results["Acc_List"].append(accuracies)
        else:
            results["Model"].append(label)
            results["Mean_Acc"].append(0)
            results["Error"].append(0)
            results["Acc_List"].append([])

    return pd.DataFrame(results)


def p_to_star(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def draw_figure(df_plot, output_path: Path):
    npg_colors = [
        "#4DBBD5",
        "#E64B35",
        "#00A087",
        "#3C5488",
        "#F39B7F",
        "#8491B4",
    ]

    colors = npg_colors[: len(df_plot)]
    plt.figure(figsize=(8, 5))

    x = np.arange(len(df_plot))
    width = 0.6

    plt.bar(
        x,
        df_plot["Mean_Acc"],
        width,
        color=colors,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
        yerr=df_plot["Error"],
        capsize=3,
        error_kw={"ecolor": "#333333", "elinewidth": 1.5, "capthick": 1.5},
    )

    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0.0, 1.01, 0.2), fontsize=UNIFIED_FONT_SIZE)
    plt.xticks(x, df_plot["Model"], fontsize=UNIFIED_FONT_SIZE - 0.5)
    plt.ylabel("Accuracy of Gemini3 ( 7-class )", fontsize=UNIFIED_FONT_SIZE)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)

    plt.tight_layout()

    baseline_name = "D.Prompt"
    baseline_idx = df_plot.index[df_plot["Model"] == baseline_name][0]
    baseline_acc = df_plot.loc[baseline_idx, "Acc_List"]
    y_max = df_plot["Mean_Acc"].max()

    for index in range(len(df_plot)):
        if index == baseline_idx:
            continue

        acc_other = df_plot.loc[index, "Acc_List"]
        _, p_value = ttest_rel(baseline_acc, acc_other)
        star = p_to_star(p_value)

        x1, x2 = x[baseline_idx], x[index]
        y = y_max + 0.035 + 0.035 * index

        plt.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], color="black", linewidth=1.5)
        plt.text(
            (x1 + x2) / 2,
            y + 0.01,
            star,
            ha="center",
            va="bottom",
            fontsize=UNIFIED_FONT_SIZE - 4,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"saved: {output_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Draw prompt/perturbation accuracy comparison for ICL experiments.")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT, help="Directory containing ICLperturbresult files.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    df_plot = load_and_calculate_accuracy(args.result_root)
    draw_figure(df_plot, args.output)


if __name__ == "__main__":
    main()
