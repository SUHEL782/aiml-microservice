from fastapi import FastAPI
from pydantic import BaseModel
from app.preprocess import preprocess
from app.model import predict

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict_api(data: InputData):
    features = preprocess(data.dict())
    result = predict(features)
    return {"prediction": result}
