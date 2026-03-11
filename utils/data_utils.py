import re
import json
import logging

logger = logging.getLogger(__name__)

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.replace("0-2", "O-2").replace("0-3", "O-3")

def safe_json_parse(text: str):
    """JSON extractor, processing the output format of large models"""
    if not text or not isinstance(text, str): return None
    clean_text = text.strip()
    match = re.search(r"```json\s*(.*?)\s*```", clean_text, re.DOTALL | re.IGNORECASE)
    if match:
        try: return json.loads(match.group(1).strip())
        except: pass
    match = re.search(r"(\[.*\]|\{.*\})", clean_text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1).strip())
        except: pass
    try: return json.loads(clean_text)
    except: return None

def extract_score(text: str, regex_pattern: str) -> str:
    """Score extraction"""
    text = text.replace("No", "NO").replace("no", "NO").replace("。", "").strip()
    m = re.search(regex_pattern, text, re.IGNORECASE)
    return m.group(1).upper() if m else ""