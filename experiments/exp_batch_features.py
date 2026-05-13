import argparse
import json
import logging
import os
import sys
from typing import Dict, Iterable, List, Tuple

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import BASE_WORKSPACE, DEFAULT_MODEL
from core.llm_client import LLMChat
from core.task_config import TASK_CONFIG
from utils.data_utils import extract_score, load_text, safe_json_parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_DATA_DIRS = {
    "A": "dataset_class5/Adata",
    "DR": "dataset_class5/DRdata",
    "H": "dataset_class5/Hdata",
    "IM": "dataset_class5/IMdata",
    "N": "dataset_class5/Ndata",
}

DATASET_ALIASES = {
    "batch": "testdataset",
    "test": "testdataset",
    "testdataset": "testdataset",
    "external": "externaldataset",
    "externaldataset": "externaldataset",
}

OUTPUT_COLUMNS = [
    "image",
    "generated_diagnosis",
    "generated_diagnosis_en",
    "raw_model_output",
    "raw_batch_response",
    "score",
    "correct",
]


def resolve_dataset_name(dataset: str) -> str:
    try:
        return DATASET_ALIASES[dataset.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unknown dataset '{dataset}'. Choose one of: {choices}") from exc


def preferred_subfolders(task_name: str, test_dir: str) -> List[str]:
    ordered = list(TASK_CONFIG[task_name]["classes"])
    if task_name == "N":
        ordered.extend(["NO", "no"])

    seen = set()
    folders = []
    for folder in ordered:
        key = folder.lower()
        if key not in seen and os.path.isdir(os.path.join(test_dir, folder)):
            folders.append(folder)
            seen.add(key)

    for folder in sorted(os.listdir(test_dir)):
        folder_path = os.path.join(test_dir, folder)
        key = folder.lower()
        if os.path.isdir(folder_path) and key not in seen:
            folders.append(folder)
            seen.add(key)
    return folders


def collect_test_images(task_name: str, dataset_name: str) -> Tuple[str, List[str], List[str]]:
    test_dir = os.path.join(BASE_WORKSPACE, dataset_name, task_name)
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")

    test_images = []
    rel_paths = []
    for subfolder in preferred_subfolders(task_name, test_dir):
        folder_path = os.path.join(test_dir, subfolder)
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(".jpg"):
                img_path = os.path.join(folder_path, fname)
                test_images.append(img_path)
                rel_paths.append(normalize_rel_path(task_name, f"{task_name}\\{os.path.relpath(img_path, test_dir)}"))
    return test_dir, test_images, rel_paths


def load_true_labels(task_name: str, dataset_name: str) -> Dict[str, str]:
    label_path = os.path.join(BASE_WORKSPACE, dataset_name, task_name, "image_scores.json")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"True-label JSON does not exist: {label_path}")
    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rel_path(task_name: str, path_text: str) -> str:
    if not path_text:
        return ""
    parts = str(path_text).replace("/", "\\").strip().split("\\")
    task_index = next((idx for idx, part in enumerate(parts) if part.upper() == task_name.upper()), None)
    if task_index is not None:
        parts = parts[task_index:]
    if parts:
        parts[0] = task_name
    return "\\".join(part for part in parts if part)


def true_label_to_class(label: str) -> str:
    if label is None:
        return ""
    label = str(label).strip().upper()
    if label == "NO":
        return "NO"
    return label


def prediction_to_class(task_name: str, score: str) -> str:
    score = str(score or "").strip().upper()
    if task_name == "A":
        if score in {"C-0", "C-1"}:
            return "0"
        if score in {"C-2", "C-3"}:
            return "1"
        if score in {"O-1", "O-2", "O-3"}:
            return "2"
        if score == "NO":
            return "NO"
    else:
        prefix = "" if task_name == "N" else f"{task_name}-"
        if task_name == "N":
            mapping = {"N-0": "0", "N-1": "1", "NO": "NO"}
        else:
            mapping = {f"{prefix}0": "0", f"{prefix}1": "1", f"{prefix}2": "2", "NO": "NO"}
        return mapping.get(score, "")
    return ""


def compute_correctness(
    task_name: str,
    image_path: str,
    predicted_score: str,
    true_labels: Dict[str, str],
) -> int:
    normalized = normalize_rel_path(task_name, image_path)
    true_label = true_labels.get(normalized.replace("\\", "/"))
    if true_label is None:
        true_label = true_labels.get(normalized)
    pred_class = prediction_to_class(task_name, predicted_score)
    true_class = true_label_to_class(true_label)
    return int(bool(pred_class) and bool(true_class) and pred_class == true_class)


def extract_task_score(task_name: str, text: str) -> str:
    if task_name == "A":
        text = str(text or "").replace("0-2", "O-2").replace("0-3", "O-3")
    return extract_score(str(text or ""), TASK_CONFIG[task_name]["regex"])


def resolve_item_path(task_name: str, item: Dict, rel_batch: List[str], idx: int) -> str:
    item_id = item.get("id")
    if isinstance(item_id, int) and 1 <= item_id <= len(rel_batch):
        return rel_batch[item_id - 1]
    if isinstance(item_id, str) and item_id.isdigit() and 1 <= int(item_id) <= len(rel_batch):
        return rel_batch[int(item_id) - 1]

    candidate = str(item.get("image", "")).strip()
    if candidate and ("\\" in candidate or "/" in candidate or candidate.upper().startswith(task_name.upper())):
        return normalize_rel_path(task_name, candidate)
    if idx < len(rel_batch):
        return rel_batch[idx]
    return ""


def append_rows(result_excel: str, rows: List[Dict]):
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if os.path.exists(result_excel):
        old_df = pd.read_excel(result_excel)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_excel(result_excel, index=False)


def run_feature_test(task_name: str, dataset: str = "testdataset", batch_size: int = 8):
    """
    Run batch or external validation testing for one KGS feature.

    dataset accepts:
    - testdataset / batch: batch testing set
    - externaldataset / external: external validation set
    """
    dataset_name = resolve_dataset_name(dataset)
    cfg = TASK_CONFIG[task_name]
    logger.info("========== Running %s on %s ==========", task_name, dataset_name)

    base_dir = os.path.join(BASE_WORKSPACE, TASK_DATA_DIRS[task_name])
    true_labels = load_true_labels(task_name, dataset_name)

    llm = LLMChat()
    llm.add_system_prompt(load_text(os.path.join(base_dir, "prompt.txt")))

    with open(os.path.join(base_dir, "scoredata/sample_picture/train_examples.json"), "r", encoding="utf-8") as f:
        llm.add_training_examples(json.load(f), os.path.join(base_dir, "imgdata/sample_picture"), cfg)

    for group_id in [1, 2]:
        fb_dir = os.path.join(base_dir, f"imgdata/data{group_id}")
        fb_json = os.path.join(base_dir, f"scoredata/data{group_id}/true_scores.json")
        with open(fb_json, "r", encoding="utf-8") as f:
            llm.add_feedback_examples(json.load(f), fb_dir)
        logger.info("Added feedback group %s", group_id)

    _, test_images, rel_paths = collect_test_images(task_name, dataset_name)

    result_excel = os.path.join(
        BASE_WORKSPACE,
        "result_batch",
        DEFAULT_MODEL,
        dataset_name,
        f"result_{task_name}.xlsx",
    )
    os.makedirs(os.path.dirname(result_excel), exist_ok=True)

    done_images = set()
    if os.path.exists(result_excel):
        old_df = pd.read_excel(result_excel)
        if "image" in old_df.columns:
            done_images = set(old_df["image"].astype(str).tolist())

    pending = [(abs_path, rel_path) for abs_path, rel_path in zip(test_images, rel_paths) if rel_path not in done_images]
    logger.info("Collected %s images, pending %s", len(test_images), len(pending))

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        abs_batch = [item[0] for item in batch]
        rel_batch = [item[1] for item in batch]
        identifiers = [str(idx) for idx in range(1, len(abs_batch) + 1)]

        model_text, raw_batch_response = llm.request_batch_assessment(
            abs_batch,
            cfg["score_range"],
            identifiers,
            output_key="id",
            return_raw=True,
        )

        parsed = safe_json_parse(model_text)
        if isinstance(parsed, dict):
            parsed = [parsed]

        rows = []
        if parsed:
            for idx, item in enumerate(parsed):
                rel_path = resolve_item_path(task_name, item, rel_batch, idx)
                final_score = extract_task_score(task_name, item.get("score", ""))
                diagnosis_text = str(item.get("reason", "")).strip()
                rows.append(
                    {
                        "image": normalize_rel_path(task_name, rel_path),
                        "generated_diagnosis": diagnosis_text,
                        "generated_diagnosis_en": diagnosis_text,
                        "raw_model_output": json.dumps(item, ensure_ascii=False),
                        "raw_batch_response": raw_batch_response,
                        "score": final_score,
                        "correct": compute_correctness(task_name, rel_path, final_score, true_labels),
                    }
                )
        else:
            rows.append(
                {
                    "image": "BATCH",
                    "generated_diagnosis": "",
                    "generated_diagnosis_en": "",
                    "raw_model_output": model_text,
                    "raw_batch_response": raw_batch_response,
                    "score": "",
                    "correct": 0,
                }
            )

        append_rows(result_excel, rows)
        logger.info("Progress: %s/%s", min(start + batch_size, len(pending)), len(pending))


def parse_args():
    parser = argparse.ArgumentParser(description="Run KGS batch-feature experiments.")
    parser.add_argument(
        "--dataset",
        default="testdataset",
        choices=sorted(DATASET_ALIASES),
        help="Choose batch test set or external validation set.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["A", "DR", "H", "IM", "N"],
        choices=["A", "DR", "H", "IM", "N"],
        help="One or more KGS features to run.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for task in args.tasks:
        run_feature_test(task_name=task, dataset=args.dataset, batch_size=args.batch_size)
