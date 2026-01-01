from fastapi import FastAPI
import joblib
import numpy as np
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/fatigue_model.pkl"
DATA_PATH = "data/processed/features.csv"

CLASS_MAP = {
    0: "Fatiguing",
    1: "Stable",
    2: "Trending"
}

# FASTAPI APP
app = FastAPI(
    title="Product Fatigue Predictor",
    description="Predicts whether a product is Trending, Stable, or Fatiguing",
    version="1.0"
)

def train_and_save_model():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model

# LOAD OR TRAIN MODEL

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_and_save_model()

# ROUTES

@app.get("/")
def home():
    return {"message": "Product Fatigue Prediction API is running"}


@app.post("/predict")
def predict(data: dict):
    features = np.array([[
        data["recent_avg_sales"],
        data["previous_avg_sales"],
        data["sales_trend"],
        data["sales_ratio"],
        data["sales_diff"],
        data["momentum_score"],
        data["decline_flag"]
    ]])

    prediction = model.predict(features)[0]
    label = CLASS_MAP[prediction]

    explanation = {
        "Fatiguing": "Product is losing sales momentum",
        "Stable": "Product performance is stable",
        "Trending": "Product demand is increasing"
    }

    return {
        "prediction": label,
        "confidence_note": explanation[label]
    }
