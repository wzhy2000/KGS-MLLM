import sys
import os
import json
import pandas as pd
import logging
from core.llm_client import LLMChat
from core.task_config import TASK_CONFIG
from config import BASE_WORKSPACE, DEFAULT_MODEL
from utils.data_utils import load_text, safe_json_parse, extract_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_feature_test(task_name: str, batch_size: int = 8):
    """
    Kyoto Gastritis 5-Category Batch Testing Experiment.
    Taskname must be one of "A", "DR", "H", "IM", "N".
    """
    cfg = TASK_CONFIG[task_name]
    logger.info(f"========== 开始执行 {task_name} 批量测试 ==========")

    folder_map = {"A": "dataset_class5/Adata", "DR": "dataset_class5/DRdata", 
                  "H": "dataset_class5/Hdata", "IM": "dataset_class5/IMdata", "N": "dataset_class5/Ndata"}
    base_dir = os.path.join(BASE_WORKSPACE, folder_map[task_name])
    
    llm = LLMChat()
    llm.add_system_prompt(load_text(os.path.join(base_dir, "prompt.txt")))
    
    with open(os.path.join(base_dir, "scoredata/sample_picture/train_examples.json"), "r", encoding="utf-8") as f:
        llm.add_training_examples(json.load(f), os.path.join(base_dir, "imgdata/sample_picture"), cfg)

    for i in [1, 2]:
        fb_dir = os.path.join(base_dir, f"imgdata/data{i}")
        with open(os.path.join(base_dir, f"scoredata/data{i}/true_scores.json"), "r", encoding="utf-8") as f:
            llm.add_feedback_examples(json.load(f), fb_dir)
        logger.info(f"已加入第{i}组反馈学习")

    test_dir = os.path.join(BASE_WORKSPACE, f"testdataset/{task_name}")
    result_excel = os.path.join(BASE_WORKSPACE, f"result_batch/{DEFAULT_MODEL}/result_{task_name}.xlsx")
    os.makedirs(os.path.dirname(result_excel), exist_ok=True)

    test_images, identifiers = [], []
    for subfolder in cfg["classes"]:
        folder_path = os.path.join(test_dir, subfolder)
        if not os.path.exists(folder_path): continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(".jpg"):
                test_images.append(os.path.join(folder_path, fname))
                ident = f"图片 {len(test_images)}"
                identifiers.append(ident)

    done_images = set()
    if os.path.exists(result_excel) and "image" in pd.read_excel(result_excel).columns:
        done_images = set(pd.read_excel(result_excel)["image"].astype(str).tolist())

    all_results = []
    for i in range(0, len(test_images), batch_size):
        batch_imgs = test_images[i:i+batch_size]
        batch_idents = identifiers[i:i+batch_size]
        
        real_rel_paths = [f"{task_name}\\{os.path.relpath(p, test_dir)}" for p in batch_imgs]
        
        if all(p in done_images for p in real_rel_paths): continue

        model_text = llm.request_batch_assessment(batch_imgs, cfg["score_range"], batch_idents)
        parsed = safe_json_parse(model_text)
        
        if parsed:
            for item, real_path in zip(parsed, real_rel_paths):
                all_results.append({
                    "image": real_path,
                    "model_output": item.get("reason", ""),
                    "score": extract_score(item.get("score", ""), cfg["regex"])
                })
        
        df = pd.DataFrame(all_results)
        if os.path.exists(result_excel):
            df = pd.concat([pd.read_excel(result_excel), df], ignore_index=True)
        df.to_excel(result_excel, index=False)
        all_results.clear()
        logger.info(f"进度: {min(i+batch_size, len(test_images))}/{len(test_images)}")

if __name__ == "__main__":
    for task in ["A", "DR", "H", "IM", "N"]:
        run_feature_test(task_name=task)