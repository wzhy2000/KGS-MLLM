import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pandas as pd
import time
import logging
from core.llm_client import LLMChat
from core.task_config import TASK_CONFIG
from config import BASE_WORKSPACE, DEFAULT_MODEL
from utils.data_utils import load_text, safe_json_parse, extract_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_cv_pipeline(batch_size: int = 4, num_groups: int = 6):
    result_root = os.path.join(BASE_WORKSPACE, f"result_re/{DEFAULT_MODEL}")
    os.makedirs(result_root, exist_ok=True)

    base_dir = os.path.join(BASE_WORKSPACE, "nshotdataset")
    prompt_path = os.path.join(base_dir, "prompt.txt")
    train_dir = os.path.join(base_dir, "imgdata/sample_picture")
    
    with open(os.path.join(base_dir, "scoredata/sample_picture/train_examples.json"), "r", encoding="utf-8") as f:
        train_examples = json.load(f)

    task_cfg = TASK_CONFIG["A"]

    for fixed_group in range(1, num_groups + 1):
        logger.info(f"===== 开始第{fixed_group}组测试实验 =====")
        llm = LLMChat()
        llm.add_system_prompt(load_text(prompt_path))
        llm.add_training_examples(train_examples, train_dir, task_cfg)

        test_dir = os.path.join(base_dir, f"imgdata/data{fixed_group}")
        test_images = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".jpg")])

        run_and_save_stage(llm, test_images, batch_size, fixed_group, 0, result_root, task_cfg)

        train_groups = [g for g in range(1, num_groups+1) if g != fixed_group]
        for idx, train_group in enumerate(train_groups, 1):
            fb_dir = os.path.join(base_dir, f"imgdata/data{train_group}")
            fb_json = os.path.join(base_dir, f"scoredata/data{train_group}/true_scores.json")
            with open(fb_json, "r", encoding="utf-8") as f:
                llm.add_feedback_examples(json.load(f), fb_dir)
            logger.info(f"已加入第{train_group}组反馈学习")
            run_and_save_stage(llm, test_images, batch_size, fixed_group, idx, result_root, task_cfg)

def run_and_save_stage(llm, test_images, batch_size, group_id, stage, result_root, task_cfg):
    results = []
    for i in range(0, len(test_images), batch_size):
        batch = test_images[i:i+batch_size]
        start_time = time.time()
        
        identifiers = [os.path.basename(p) for p in batch]
        model_text = llm.request_batch_assessment(batch, task_cfg["score_range"], identifiers)
        elapsed = time.time() - start_time

        parsed = safe_json_parse(model_text)
        if parsed:
            for item, orig_path in zip(parsed, batch):
                results.append({
                    "stage": stage,
                    "image": f"data{group_id}_{os.path.basename(orig_path)}",
                    "model_output": item.get("reason", ""),
                    "score": extract_score(item.get("score", ""), task_cfg["regex"]),
                    "response_time": elapsed / len(parsed)
                })
        else:
            results.append({"stage": stage, "image": "BATCH", "score": ""})
            
    out_path = os.path.join(result_root, f"re_fixed{group_id}_stage{stage}.xlsx")
    pd.DataFrame(results).to_excel(out_path, index=False)

if __name__ == "__main__":
    run_cv_pipeline()