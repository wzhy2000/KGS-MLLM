import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pandas as pd
import logging
from core.llm_client import LLMChat
from core.task_config import TASK_CONFIG
from config import BASE_WORKSPACE, DEFAULT_MODEL
from utils.data_utils import load_text, safe_json_parse, extract_score
from utils.image_utils import rotate_image, augment_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_perturbation(condition: str, batch_size: int = 4):
    """
    Perform perturbation experiments (under a total of 6 conditions).
    Optional conditions:
    -'baseline': Detailed structured prompt words (D.Prompt) ->corresponding prompt_all.txt
    -'short_prompt': Brief prompt word (S.Prompt) ->corresponding prompt_e.txt
    -'rev_cat ': Flip the order of categories within the prompt word (Rev.Cat.) ->Corresponding prompt_R.txt
    -'rev_exp ': Flip the order of examples within the prompt word (Rev.Exp.) ->corresponding sample_picture_R
    -'rotated': Test image rotated (Rotated)
    -'aug': Test image random enhancement (Aug.)
        
    It must run under the condition of 14 shot (basic 2-shot+data1~data3).
    """
    logger.info(f"--- 启动扰动实验: {condition} ---")
    base_dir = os.path.join(BASE_WORKSPACE, "ICLA25/endodata")
    
    # Configure prompt words and example source paths based on the condition
    if condition == 'short_prompt':
        prompt_file = "prompt_e.txt"
    elif condition == 'rev_cat':
        prompt_file = "prompt_R.txt"
    else:
        prompt_file = "prompt_all.txt"
        
    train_folder = "sample_picture_R" if condition == 'rev_exp' else "sample_picture"
    
    prompt_path = os.path.join(base_dir, prompt_file)
    train_dir = os.path.join(base_dir, f"imgdata/{train_folder}")
    train_json = os.path.join(base_dir, f"scoredata/{train_folder}/train_examples.json")
    
    test_dir = os.path.join(base_dir, "imgdata/data_test")
    test_images = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".jpg")])

    result_excel = os.path.join(BASE_WORKSPACE, f"ICLA25/result/result_{condition}.xlsx")
    os.makedirs(os.path.dirname(result_excel), exist_ok=True)
    task_cfg = TASK_CONFIG["A"]

    # Initialize the model and load the basic 2-shot examples
    llm = LLMChat(model=DEFAULT_MODEL)
    llm.add_system_prompt(load_text(prompt_path))
    
    with open(train_json, "r", encoding="utf-8") as f:
        llm.add_training_examples(json.load(f), train_dir, task_cfg)
    logger.info(f"已加载基础 2-shot 示例 (Prompt: {prompt_file}, Data: {train_folder})")

    # Add feedback learning (data1, data2, data3) to achieve a 14 shot peak state
    for stage in [1, 2, 3]:
        fb_dir = os.path.join(base_dir, f"imgdata/data{stage}")
        fb_json = os.path.join(base_dir, f"scoredata/data{stage}/true_scores.json")
        
        with open(fb_json, "r", encoding="utf-8") as f:
            fb_scores = json.load(f)
            
        llm.add_feedback_examples(fb_scores, fb_dir)
        logger.info(f"已加入 data{stage} 作为反馈学习，当前状态为 {(stage * 4) + 2}-shot")

    done_images = set()
    if os.path.exists(result_excel) and "image" in pd.read_excel(result_excel).columns:
        done_images = set(pd.read_excel(result_excel)["image"].astype(str).tolist())
        logger.info(f"发现已处理的 {len(done_images)} 张图片，将执行断点续跑")

    all_results = []
    for i in range(0, len(test_images), batch_size):
        batch = test_images[i:i+batch_size]
        proc_paths, identifiers = [], []

        for img_path in batch:
            rel_name = os.path.basename(img_path)
            
            if condition == 'rotated':
                aug_dir = os.path.join(base_dir, "imgdata/data_rot_aug")
                os.makedirs(aug_dir, exist_ok=True)
                new_path, log = rotate_image(img_path, aug_dir)
                proc_paths.append(new_path)
                identifiers.append(f"{rel_name} (增强:{log})")
            elif condition == 'aug':
                aug_dir = os.path.join(base_dir, "imgdata/data_test_aug")
                os.makedirs(aug_dir, exist_ok=True)
                new_path, log = augment_image(img_path, os.path.join(aug_dir, rel_name))
                proc_paths.append(new_path)
                identifiers.append(f"{rel_name} (增强:{log})")
            else:
                proc_paths.append(img_path)
                identifiers.append(rel_name)

        if all(os.path.basename(p) in done_images for p in batch):
            continue

        model_text = llm.request_batch_assessment(proc_paths, task_cfg["score_range"], identifiers)
        parsed = safe_json_parse(model_text)

        if parsed:
            for item, ident in zip(parsed, identifiers):
                base_img_name = item.get("image", ident).split(" (")[0] 
                all_results.append({
                    "image": base_img_name,
                    "condition": condition,
                    "model_output": item.get("reason", ""),
                    "score": extract_score(item.get("score", ""), task_cfg["regex"])
                })
        else:
             for ident in identifiers:
                 base_img_name = ident.split(" (")[0]
                 all_results.append({"image": base_img_name, "condition": condition, "score": ""})
        
        df = pd.DataFrame(all_results)
        if os.path.exists(result_excel):
            df = pd.concat([pd.read_excel(result_excel), df], ignore_index=True)
        df.to_excel(result_excel, index=False)
        all_results.clear()
        
        logger.info(f"[{condition}] 批次完成，当前进度: {min(i+batch_size, len(test_images))}/{len(test_images)}")

if __name__ == "__main__":
    all_conditions = ['baseline', 'short_prompt', 'rev_cat', 'rev_exp', 'rotated', 'aug']
    for cond in all_conditions:
        run_perturbation(cond, batch_size=8)

        