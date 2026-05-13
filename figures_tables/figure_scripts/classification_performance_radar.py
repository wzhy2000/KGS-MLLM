from __future__ import annotations

import argparse
import re
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TABLE_DIR = Path(__file__).resolve().parents[1] / "table"
IMAGE_DIR = Path(__file__).resolve().parents[1] / "image"

MODEL_REPORT = TABLE_DIR / "model_group_average_report.txt"
DOCTOR_REPORT = TABLE_DIR / "doctor_group_average_report.txt"
DEFAULT_OUTPUT = IMAGE_DIR / "classification_performance_radar.png"

TITLE = (
    "Classification Performance of MLLMs, DLKGs, and Endoscopists "
    "Across Five Endoscopic Findings"
)

MODEL_MAPPING = {
    "result_gemini": "Gemini3",
    "result_gpt": "GPT5.2",
    "result_claude": "Claude4.5",
    "result_grok": "Grok4",
    "result_DLKGS": "DLKGs",
}

DOCTOR_MAPPING = {
    "group_1_doctor1_5": "Senior",
    "group_2_doctor6_10": "Junior",
    "groups_3": "Mixed",
}

MODEL_ORDER = ["Gemini3", "GPT5.2", "Claude4.5", "Grok4", "DLKGs", "Senior", "Junior", "Mixed"]
METRICS = ["Accuracy (%)", "Specificity (%)", "Precision (%)", "Recall (%)", "F1 score"]
DISPLAY_METRICS = ["Accuracy", "Specificity", "Precision", "Recall", "F1 score"]
CATEGORIES = ["A", "DR", "H", "IM", "N", "overall"]

COLORS = [
    "#4DBBD5",
    "#E64B35",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
]


def parse_report(report_path: Path, block_mapping: dict[str, str]) -> list[dict[str, float | str]]:
    text = report_path.read_text(encoding="utf-8")
    rows: list[dict[str, float | str]] = []
    current_block: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            current_block = line[1:-1]
            continue

        if current_block is None:
            continue

        if line.startswith("Sub-category") or line.startswith("="):
            continue

        if line.startswith(("A-all", "DR-all", "H-all", "IM-all", "N-all", "overall")):
            if current_block not in block_mapping:
                continue

            parts = re.split(r"\s+", line)
            metric_name = parts[0]
            category = metric_name.replace("-all", "") if metric_name != "overall" else "overall"
            rows.append(
                {
                    "Category": category,
                    "Model": block_mapping[current_block],
                    "Accuracy (%)": float(parts[2].rstrip("%")),
                    "Specificity (%)": float(parts[3].rstrip("%")),
                    "Precision (%)": float(parts[4].rstrip("%")),
                    "Recall (%)": float(parts[5].rstrip("%")),
                    "F1 score": float(parts[6].rstrip("%")),
                }
            )

    return rows


def load_performance_table(model_report: Path = MODEL_REPORT, doctor_report: Path = DOCTOR_REPORT) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    rows.extend(parse_report(model_report, MODEL_MAPPING))
    rows.extend(parse_report(doctor_report, DOCTOR_MAPPING))

    df = pd.DataFrame(rows)
    if df["F1 score"].max() <= 1.0:
        df["F1 score"] = df["F1 score"] * 100
    return df


def draw_radar_on_ax(ax: plt.Axes, df: pd.DataFrame, category_name: str, font_size: int) -> tuple[list, list[str]]:
    category_data = df[df["Category"] == category_name]

    metric_count = len(METRICS)
    angles = [n / float(metric_count) * 2 * pi for n in range(metric_count)]
    angles += angles[:1]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks([])
    for angle, label in zip(angles[:-1], DISPLAY_METRICS):
        angle_deg = np.degrees(angle)
        rotation = -angle_deg
        if 90 < angle_deg < 270:
            rotation += 180

        ax.text(
            angle,
            114,
            label,
            size=font_size,
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
        )

    ax.set_rlabel_position(0)
    ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
    ax.set_yticklabels(["40", "50", "60", "70", "80", "90", "100"])
    ax.set_ylim(40, 100)

    for angle in angles[:-1]:
        ax.plot([angle, angle], [40, 100], color="gray", linewidth=1.5, alpha=0.8)

    handles = []
    labels = []
    for idx, model_name in enumerate(MODEL_ORDER):
        row = category_data[category_data["Model"] == model_name]
        if row.empty:
            continue

        values = row[METRICS].values.flatten().tolist()
        values += values[:1]
        handle = ax.plot(angles, values, linewidth=2.5, label=model_name, color=COLORS[idx])
        handles.append(handle[0])
        labels.append(model_name)

    ax.text(
        0.0,
        1.05,
        category_name,
        transform=ax.transAxes,
        fontsize=font_size,
        ha="left",
        va="bottom",
    )

    return handles, labels


def plot_radar(df: pd.DataFrame, output_path: Path, show_title: bool = False) -> None:
    font_size = 26
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = font_size

    fig, axs = plt.subplots(2, 3, figsize=(22, 14), subplot_kw={"polar": True})
    all_handles = []
    all_labels = []

    for ax, category in zip(axs.flatten(), CATEGORIES):
        handles, labels = draw_radar_on_ax(ax, df, category, font_size)
        all_handles.extend(handles)
        all_labels.extend(labels)

    unique_legend = dict(zip(all_labels, all_handles))
    fig.legend(
        unique_legend.values(),
        unique_legend.keys(),
        loc="upper center",
        ncol=4,
        fontsize=font_size,
        frameon=False,
    )

    if show_title:
        fig.suptitle(TITLE, y=0.99, fontsize=font_size)
        layout_rect = [0, 0.08, 1, 0.89]
    else:
        layout_rect = [0, 0.08, 1, 0.92]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=layout_rect)
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
    df = load_performance_table(args.model_report, args.doctor_report)
    plot_radar(df, args.output, show_title=args.show_title)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
