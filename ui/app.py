import streamlit as st
import requests

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "https://product-fatigue-api.onrender.com/predict"

st.set_page_config(page_title="Product Fatigue Predictor", layout="centered")

st.title("📦 Product Fatigue Predictor")
st.write("Predict whether a product is **Trending**, **Stable**, or **Fatiguing** based on sales behavior.")

st.divider()

# -----------------------------
# INPUTS
# -----------------------------
recent_avg_sales = st.number_input("Recent Average Sales", min_value=0.0, step=0.1)
previous_avg_sales = st.number_input("Previous Average Sales", min_value=0.0, step=0.1)
sales_trend = st.number_input("Sales Trend", step=0.1)
sales_ratio = st.number_input("Sales Ratio", step=0.1)
sales_diff = st.number_input("Sales Difference", step=0.1)
momentum_score = st.number_input("Momentum Score", step=0.1)
decline_flag = st.selectbox("Decline Flag", [0, 1])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Product Status"):
    payload = {
        "recent_avg_sales": recent_avg_sales,
        "previous_avg_sales": previous_avg_sales,
        "sales_trend": sales_trend,
        "sales_ratio": sales_ratio,
        "sales_diff": sales_diff,
        "momentum_score": momentum_score,
        "decline_flag": decline_flag
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        st.success(f"Prediction: **{result['prediction']}**")
        st.info(result["confidence_note"])

    except Exception as e:
        st.error("Could not connect to prediction API")
        st.write(e)
