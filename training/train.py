import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "data/sample_train.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop("label", axis=1)
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Best pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=2,
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Best Model Accuracy: {accuracy:.2f}")

# Save
joblib.dump(model, MODEL_PATH)
print(f"Best model saved: {MODEL_PATH}")
