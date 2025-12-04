# service/app/model.py
import os
from joblib import load
from threading import Lock

_MODEL = None
_LOCK = Lock()

def load_model(path: str = None):
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            p = path or os.getenv("MODEL_PATH", "/app/models/text_classifier.joblib")
            _MODEL = load(p)
    return _MODEL
