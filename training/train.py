# training/train.py
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_train.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    return df["text"].values, df["label"].values

def train_and_save():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    param_grid = {"clf__C": [0.1, 1.0, 5.0]}
    search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    print("Best params:", search.best_params_)
    preds = search.predict(X_test)
    print(classification_report(y_test, preds))

    out_path = os.path.join(OUT_DIR, "text_classifier.joblib")
    joblib.dump(search.best_estimator_, out_path)
    print("Saved model:", out_path)

if __name__ == "__main__":
    train_and_save()
