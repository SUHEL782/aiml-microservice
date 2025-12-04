# service/app/preprocess.py
import re
from typing import List

def simple_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_batch(texts: List[str]) -> List[str]:
    return [simple_clean(t) for t in texts]
