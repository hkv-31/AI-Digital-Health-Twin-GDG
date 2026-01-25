# gemini_extractor.py
import json
import os
from pyexpat import model
import re
import google.generativeai as genai
from ai.prompts import EXTRACTION_PROMPT, TARGET_FEATURES_DEFAULT
from gemini_client import get_text_model

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def _safe_json_load(text: str) -> dict:
    """
    Gemini sometimes returns stray text. Extract the first {...} block and parse.
    """
    if not text:
        return {}
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting JSON object from within text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    candidate = m.group(0)

    try:
        return json.loads(candidate)
    except Exception:
        return {}

def extract_labs(raw_text: str, target_features=None) -> dict:
    target_features = target_features or TARGET_FEATURES_DEFAULT

    from gemini_client import get_text_model

    model = get_text_model()

    prompt = EXTRACTION_PROMPT.format(
        text=(raw_text or "")[:6000],
        target_features=json.dumps(target_features, indent=2)
    )

    response = model.generate_content(prompt)
    return _safe_json_load(getattr(response, "text", ""))
