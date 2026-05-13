import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pandas as pd
import logging
import time
from core.llm_client import LLMChat
from core.task_config import TASK_CONFIG
from config import BASE_WORKSPACE, DEFAULT_MODEL
from utils.data_utils import load_text, safe_json_parse, extract_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_nshot_pipeline(batch_size: int = 4):
    """
    Marginal diminishing returns experiment:
    By gradually adding 5 sets of image data as ICL examples (Feedback),
    Test the performance changes of the model from 2-shot to 22 shot on a fixed validation set.
    """
    result_root = os.path.join(BASE_WORKSPACE, f"result_nshot/{DEFAULT_MODEL}")
    os.makedirs(result_root, exist_ok=True)

    # Basic path configuration
    base_dir = os.path.join(BASE_WORKSPACE, "nshotdataset")
    prompt_path = os.path.join(base_dir, "prompt.txt")
    train_dir = os.path.join(base_dir, "imgdata/sample_picture")
    train_json = os.path.join(base_dir, "scoredata/sample_picture/train_examples.json")
    
    # Fixed test set
    test_dir = os.path.join(base_dir, "imgdata/data6")
    test_images = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".jpg")])

    task_cfg = TASK_CONFIG["A"]

    # Initialize model, add base prompt and 2-shot examples (14 images)
    llm = LLMChat()
    llm.add_system_prompt(load_text(prompt_path))
    
    with open(train_json, "r", encoding="utf-8") as f:
        train_examples = json.load(f)
    llm.add_training_examples(train_examples, train_dir, task_cfg)
    logger.info("已加载基础 2-shot 示例")

    # The mapping relationship between stage and n-shot
    stage_to_shot = {0: 2, 1: 6, 2: 10, 3: 14, 4: 18, 5: 22}

    # Loop through stages 0~5
    for stage in range(6):
        n_shot = stage_to_shot[stage]
        logger.info(f"========== 开始评估 {n_shot}-shot (Stage {stage}) ==========")
        
        # When Stage>0, dynamically add the newly added dataX to the context (auto accumulate)
        if stage > 0:
            fb_dir = os.path.join(base_dir, f"imgdata/data{stage}")
            fb_json = os.path.join(base_dir, f"scoredata/data{stage}/true_scores.json")
            with open(fb_json, "r", encoding="utf-8") as f:
                fb_scores = json.load(f)
            
            # Feedback adding method, each added group is equivalent to +4 shot
            llm.add_feedback_examples(fb_scores, fb_dir)
            logger.info(f"已将 data{stage} 加入上下文，当前升级为 {n_shot}-shot")

        out_excel_path = os.path.join(result_root, f"test_data6_{stage}.xlsx")
        
        # Break point and continue running
        if os.path.exists(out_excel_path) and len(pd.read_excel(out_excel_path)) >= len(test_images):
            logger.info(f"{n_shot}-shot 已经跑完，跳过测试步骤...")
            continue

        # Batch reasoning
        all_results = []
        for i in range(0, len(test_images), batch_size):
            batch = test_images[i:i+batch_size]
            identifiers = [os.path.basename(p) for p in batch]
            
            start_time = time.time()
            model_text = llm.request_batch_assessment(batch, task_cfg["score_range"], identifiers)
            elapsed = time.time() - start_time

            parsed = safe_json_parse(model_text)
            if parsed:
                for item, ident in zip(parsed, identifiers):
                    all_results.append({
                        "image": item.get("image", ident),
                        "model_output": item.get("reason", ""),
                        "score": extract_score(item.get("score", ""), task_cfg["regex"]),
                        "response_time": elapsed / len(parsed) 
                    })
            else:
                for ident in identifiers:
                    all_results.append({
                        "image": ident, 
                        "model_output": "PARSE_ERROR", 
                        "score": "", 
                        "response_time": elapsed / len(identifiers)
                    })
            
            pd.DataFrame(all_results).to_excel(out_excel_path, index=False)
            logger.info(f"{n_shot}-shot 进度: {min(i+batch_size, len(test_images))}/{len(test_images)}")
            
        logger.info(f"✔ {n_shot}-shot (Stage {stage}) 测试完毕！保存至: {out_excel_path}\n")

if __name__ == "__main__":
    run_nshot_pipeline(batch_size=8)
