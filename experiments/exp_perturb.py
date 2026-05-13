import json
import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import BASE_WORKSPACE, DEFAULT_MODEL
from core.llm_client import LLMChat
from core.task_config import TASK_CONFIG
from utils.data_utils import extract_score, load_text, safe_json_parse
from utils.image_utils import augment_image, rotate_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_perturbation(condition: str, batch_size: int = 4):
    """
    Perturbation experiment at the 14-shot setting.

    Conditions:
    - baseline: detailed structured prompt
    - short_prompt: brief prompt
    - rev_cat: reversed category order in prompt
    - rev_exp: reversed example order
    - rotated: rotated test images, with an angle column
    - aug: augmented test images, with a transform column
    """
    logger.info("--- Starting perturbation experiment: %s ---", condition)
    base_dir = os.path.join(BASE_WORKSPACE, "ICLA25/endodata")

    if condition == "short_prompt":
        prompt_file = "prompt_e.txt"
    elif condition == "rev_cat":
        prompt_file = "prompt_R.txt"
    else:
        prompt_file = "prompt_all.txt"

    train_folder = "sample_picture_R" if condition == "rev_exp" else "sample_picture"

    prompt_path = os.path.join(base_dir, prompt_file)
    train_dir = os.path.join(base_dir, f"imgdata/{train_folder}")
    train_json = os.path.join(base_dir, f"scoredata/{train_folder}/train_examples.json")

    test_dir = os.path.join(base_dir, "imgdata/data_test")
    test_images = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".jpg")])

    result_excel = os.path.join(BASE_WORKSPACE, f"ICLA25/result/result_{condition}.xlsx")
    os.makedirs(os.path.dirname(result_excel), exist_ok=True)
    task_cfg = TASK_CONFIG["A"]

    llm = LLMChat(model=DEFAULT_MODEL)
    llm.add_system_prompt(load_text(prompt_path))

    with open(train_json, "r", encoding="utf-8") as f:
        llm.add_training_examples(json.load(f), train_dir, task_cfg)
    logger.info("Loaded base examples (prompt=%s, data=%s)", prompt_file, train_folder)

    for stage in [1, 2, 3]:
        fb_dir = os.path.join(base_dir, f"imgdata/data{stage}")
        fb_json = os.path.join(base_dir, f"scoredata/data{stage}/true_scores.json")
        with open(fb_json, "r", encoding="utf-8") as f:
            llm.add_feedback_examples(json.load(f), fb_dir)
        logger.info("Added feedback data%s", stage)

    done_images = set()
    if os.path.exists(result_excel):
        old_df = pd.read_excel(result_excel)
        if "image" in old_df.columns:
            done_images = set(old_df["image"].astype(str).tolist())

    all_results = []
    for i in range(0, len(test_images), batch_size):
        batch = test_images[i : i + batch_size]
        proc_paths, identifiers, meta_logs = [], [], []

        for img_path in batch:
            rel_name = os.path.basename(img_path)
            if condition == "rotated":
                aug_dir = os.path.join(base_dir, "imgdata/data_rot_aug")
                os.makedirs(aug_dir, exist_ok=True)
                new_path, log = rotate_image(img_path, aug_dir)
                proc_paths.append(new_path)
                identifiers.append(f"{rel_name} ({log})")
                meta_logs.append(log)
            elif condition == "aug":
                aug_dir = os.path.join(base_dir, "imgdata/data_test_aug")
                os.makedirs(aug_dir, exist_ok=True)
                new_path, log = augment_image(img_path, os.path.join(aug_dir, rel_name))
                proc_paths.append(new_path)
                identifiers.append(f"{rel_name} ({log})")
                meta_logs.append(log)
            else:
                proc_paths.append(img_path)
                identifiers.append(rel_name)
                meta_logs.append("")

        if all(os.path.basename(p) in done_images for p in batch):
            continue

        model_text = llm.request_batch_assessment(proc_paths, task_cfg["score_range"], identifiers)
        parsed = safe_json_parse(model_text)

        if parsed:
            for item, ident, meta_log in zip(parsed, identifiers, meta_logs):
                base_img_name = item.get("image", ident).split(" (")[0]
                row = {
                    "image": base_img_name,
                    "model_output": item.get("reason", ""),
                    "score": extract_score(item.get("score", ""), task_cfg["regex"]),
                }
                if condition == "rotated":
                    row["angle"] = meta_log
                elif condition == "aug":
                    row["transform"] = meta_log
                all_results.append(row)
        else:
            for ident, meta_log in zip(identifiers, meta_logs):
                row = {
                    "image": ident.split(" (")[0],
                    "model_output": "PARSE_ERROR",
                    "score": "",
                }
                if condition == "rotated":
                    row["angle"] = meta_log
                elif condition == "aug":
                    row["transform"] = meta_log
                all_results.append(row)

        df = pd.DataFrame(all_results)
        if os.path.exists(result_excel):
            df = pd.concat([pd.read_excel(result_excel), df], ignore_index=True)
        df.to_excel(result_excel, index=False)
        all_results.clear()

        logger.info("[%s] progress: %s/%s", condition, min(i + batch_size, len(test_images)), len(test_images))


if __name__ == "__main__":
    all_conditions = ["baseline", "short_prompt", "rev_cat", "rev_exp", "rotated", "aug"]
    for cond in all_conditions:
        run_perturbation(cond, batch_size=8)
