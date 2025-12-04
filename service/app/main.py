# service/app/main.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from .model import load_model
from .preprocess import preprocess_batch
from .auth import require_api_key
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    predictions: List[str]
    confidences: List[float]

app = FastAPI(title="AIML Inference Service", version="1.0.0")

@app.on_event("startup")
def startup():
    model_path = os.getenv("MODEL_PATH", "/app/models/text_classifier.joblib")
    logger.info(f"Loading model from {model_path}")
    load_model(model_path)
    logger.info("Model loaded")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, ok: bool = Depends(require_api_key)):
    model = load_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    texts = preprocess_batch(req.texts)
    preds = model.predict(texts)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
        confidences = probs.max(axis=1).tolist()
    else:
        confidences = [1.0] * len(preds)
    return {"predictions": preds.tolist(), "confidences": confidences}

@app.get("/info")
def info():
    m = load_model()
    return {"model_class": str(type(m)), "model_path": os.getenv("MODEL_PATH")}
