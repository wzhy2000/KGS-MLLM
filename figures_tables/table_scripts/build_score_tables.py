from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "result" / "testdataset_doctorscore"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "table"

MODEL_DIRS = {
    "Gemini": "1.result_gemini",
    "GPT": "2.result_gpt",
    "Claude": "3.result_claude",
    "Grok": "4.result_grok",
}


def collect_average_values(folder: Path, correct_value: int) -> list[float]:
    values: list[float] = []
    for path in sorted(folder.glob("*.xlsx")):
        df = pd.read_excel(path, usecols=["Average", "correct"])
        subset = df[df["correct"].astype(int) == correct_value]["Average"].tolist()
        values.extend(subset)
    return values


def build_table(source_root: Path, correct_value: int) -> pd.DataFrame:
    data = {
        model_name: pd.Series(collect_average_values(source_root / folder, correct_value), dtype="float64")
        for model_name, folder in MODEL_DIRS.items()
    }
    return pd.DataFrame(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build score0/score1 tables from doctor-score result workbooks.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    score0_df = build_table(args.source_root, correct_value=0)
    score1_df = build_table(args.source_root, correct_value=1)

    score0_output = args.output_dir / "score0.xlsx"
    score1_output = args.output_dir / "score1.xlsx"
    score0_df.to_excel(score0_output, index=False)
    score1_df.to_excel(score1_output, index=False)

    print(f"saved: {score0_output}")
    print(f"saved: {score1_output}")
    print(f"score0 rows: {len(score0_df)}")
    print(f"score1 rows: {len(score1_df)}")


if __name__ == "__main__":
    main()
