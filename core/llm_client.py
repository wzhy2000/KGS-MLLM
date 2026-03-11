import time
import requests
import logging
import os
from typing import List, Dict, Any
from config import API_KEY, PROXY_API_URL, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS
from utils.image_utils import encode_image_to_data_url, safe_join

logger = logging.getLogger(__name__)

class LLMChat:
    def __init__(self, model=DEFAULT_MODEL):
        self.model = model
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}',
        }
        self.conversation_history: List[Dict[str, Any]] = []

    def add_system_prompt(self, prompt: str):
        self.conversation_history.append({'role': 'system', 'content': prompt})

    def add_training_examples(self, train_examples: List[Dict], img_folder: str, task_cfg: Dict):
        for ex in train_examples:
            img_path = safe_join(img_folder, ex["image"])
            if not os.path.exists(img_path): continue
            
            description = task_cfg["format_example"](ex)
            score_clean = ex['score'].replace("图片评分：", "").replace("。", "").replace("（无表现）", "").strip().upper()
            
            self.conversation_history.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"示例学习：请学习此图像的观察描述与评分。\n{description}"},
                    {"type": "image_url", "image_url": {"url": encode_image_to_data_url(img_path)}}
                ]
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": f"该示例的评分为 {score_clean}，理由：{task_cfg['reasoning']}"
            })

    def add_feedback_examples(self, feedback_scores: Dict[str, str], img_folder: str):
        for fname, true_score in feedback_scores.items():
            img_path = safe_join(img_folder, fname)
            if not os.path.exists(img_path): continue
            
            true_score_clean = true_score.replace("（无表现）", "").strip().upper()
            self.conversation_history.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"反馈学习：这张图像的真实评分是 {true_score_clean}。"},
                    {"type": "image_url", "image_url": {"url": encode_image_to_data_url(img_path)}}
                ]
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": f"我已记录这张图像的真实评分为 {true_score_clean}，并将其纳入后续判定参考。"
            })

    def request_batch_assessment(self, img_paths: List[str], score_range: str, identifiers: List[str]) -> str:
        contents = [{"type": "text", "text": (
            "请逐张分析以下测试图片，并按编号输出结果。\n"
            "输出格式要求：JSON 数组，每个元素包含：\n"
            "{'image': 标识, 'score': 评分, 'reason': 分析理由}\n"
            f"评分范围：{score_range}"
        )}]

        for img_path, identifier in zip(img_paths, identifiers):
            img_data_url = encode_image_to_data_url(img_path)
            contents.append({"type": "text", "text": f"标识: {identifier}"})
            contents.append({"type": "image_url", "image_url": {"url": img_data_url}})

        data = {
            'model': self.model,
            'messages': self.conversation_history + [{"role": "user", "content": contents}],
            'temperature': DEFAULT_TEMPERATURE,
            'top_p': DEFAULT_TOP_P,
            'max_tokens': DEFAULT_MAX_TOKENS,
        }

        for attempt in range(3):
            try:
                response = requests.post(PROXY_API_URL, headers=self.headers, json=data, timeout=120)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except Exception as e:
                logger.warning(f"请求失败 (尝试 {attempt+1}/3): {e}")
                time.sleep(2)
        return "请求失败"