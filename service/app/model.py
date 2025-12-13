import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "../../training/models/best_model.pkl"
)

model = joblib.load(MODEL_PATH)

def predict(features):
    return int(model.predict([features])[0])
