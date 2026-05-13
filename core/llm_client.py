import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple, Union

import requests

from config import (
    API_KEY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    PROXY_API_URL,
)
from utils.image_utils import encode_image_to_data_url, safe_join

logger = logging.getLogger(__name__)


class LLMChat:
    def __init__(self, model=DEFAULT_MODEL):
        self.model = model
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        self.conversation_history: List[Dict[str, Any]] = []

    def add_system_prompt(self, prompt: str):
        self.conversation_history.append({"role": "system", "content": prompt})

    def add_training_examples(self, train_examples: List[Dict], img_folder: str, task_cfg: Dict):
        for ex in train_examples:
            img_path = safe_join(img_folder, ex["image"])
            if not os.path.exists(img_path):
                continue

            description = task_cfg["format_example"](ex)
            score_clean = (
                ex["score"]
                .replace("图片评分：", "")
                .replace("。", "")
                .replace("（无表现）", "")
                .strip()
                .upper()
            )

            self.conversation_history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Example learning: learn this image description and score.\n{description}",
                        },
                        {"type": "image_url", "image_url": {"url": encode_image_to_data_url(img_path)}},
                    ],
                }
            )
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": f"The score of this example is {score_clean}. Reason: {task_cfg['reasoning']}",
                }
            )

    def add_feedback_examples(self, feedback_scores: Dict[str, str], img_folder: str):
        for fname, true_score in feedback_scores.items():
            img_path = safe_join(img_folder, fname)
            if not os.path.exists(img_path):
                continue

            true_score_clean = true_score.replace("（无表现）", "").strip().upper()
            self.conversation_history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Feedback learning: the true score is {true_score_clean}."},
                        {"type": "image_url", "image_url": {"url": encode_image_to_data_url(img_path)}},
                    ],
                }
            )
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": (
                        f"I have recorded the true score as {true_score_clean} "
                        "and will use it as reference for subsequent judgments."
                    ),
                }
            )

    def request_batch_assessment(
        self,
        img_paths: List[str],
        score_range: str,
        identifiers: List[str],
        output_key: str = "image",
        return_raw: bool = False,
    ) -> Union[str, Tuple[str, str]]:
        """
        Request a batch prediction.

        output_key="image" keeps the original compact interface used by other
        experiment scripts. output_key="id" matches the LLM_X_ALL.py batch
        scripts, where each output item is keyed by batch-local id.
        """
        id_field = "id" if output_key == "id" else "image"
        id_label = "number" if output_key == "id" else "identifier"
        contents = [
            {
                "type": "text",
                "text": (
                    "Please analyze the following test images one by one and return results by number.\n"
                    "Output format: a JSON array. Each element must contain:\n"
                    f"{{'{id_field}': {id_label}, 'score': score, 'reason': analysis_reason}}\n"
                    f"Score range: {score_range}"
                ),
            }
        ]

        for idx, (img_path, identifier) in enumerate(zip(img_paths, identifiers), 1):
            label = f"Image {idx}" if output_key == "id" else f"Identifier: {identifier}"
            contents.append({"type": "text", "text": label})
            contents.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(img_path)}})

        data = {
            "model": self.model,
            "messages": self.conversation_history + [{"role": "user", "content": contents}],
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }

        for attempt in range(3):
            try:
                response = requests.post(PROXY_API_URL, headers=self.headers, json=data, timeout=150)
                response.raise_for_status()
                response_json = response.json()
                model_text = response_json["choices"][0]["message"]["content"]
                if return_raw:
                    return model_text, json.dumps(response_json, ensure_ascii=False)
                return model_text
            except Exception as e:
                logger.warning("Request failed (attempt %s/3): %s", attempt + 1, e)
                time.sleep(2)

        if return_raw:
            return "REQUEST_FAILED", ""
        return "REQUEST_FAILED"
