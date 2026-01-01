from fastapi import FastAPI
import joblib
import numpy as np

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "models/fatigue_model.pkl"

model = joblib.load(MODEL_PATH)

# Class mapping
CLASS_MAP = {
    0: "Fatiguing",
    1: "Stable",
    2: "Trending"
}

app = FastAPI(
    title="Product Fatigue Predictor",
    description="Predicts whether a product is Trending, Stable, or Fatiguing",
    version="1.0"
)


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

    message = {
        "Fatiguing": "Product is losing sales momentum",
        "Stable": "Product performance is stable",
        "Trending": "Product demand is increasing"
    }

    return {
        "prediction": label,
        "confidence_note": message[label]
    }
