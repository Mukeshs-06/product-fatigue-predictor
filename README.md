# 📦 Product Fatigue Predictor

> An end-to-end Machine Learning project that detects whether a product is **Trending**, **Stable**, or **Fatiguing** based on sales behavior — from raw data to a deployed UI.

---

## 👋 What is this project about?

In real businesses, one common question is:

> *“Is this product still doing well, or is it slowly losing customer interest?”*

This project answers that question using **historical sales data**.

I built a complete ML system that:

* Analyzes product sales over time
* Detects sales momentum (upward, stable, or declining)
* Classifies products into:

  * 🔥 **Trending**
  * 😐 **Stable**
  * 💤 **Fatiguing**
* Exposes predictions through a **cloud API**
* Provides a **simple UI** for interaction

This is not just a model — it’s a **working ML product**.

---

## 🧠 Why this project is interesting

Most datasets **do not come with labels** like *“this product is fatiguing”*.
So instead of relying on pre-labeled data, this project:

* **Creates labels using business logic**
* Mimics how real teams think about product performance
* Focuses on **interpretability and realism**, not just accuracy

This makes the project practical and interview-ready.

---

## 📂 Dataset

* Transaction-level ecommerce sales data
* Each row represents an order
* Key columns used:

  * `SKU` – product identifier
  * `Date` – order date
  * `Qty` – units sold
  * `Amount` – revenue

No ratings or reviews were used — product fatigue is inferred **purely from sales momentum**, which is very common in real businesses.

---

## 🏷️ Label Engineering (Core Idea)

Since fatigue labels don’t naturally exist, I **engineered them**.

For each product (`SKU`):

1. Aggregate daily sales
2. Create a continuous time series (missing days → zero sales)
3. Compare:

   * **Recent 30-day average sales**
   * **Previous 30-day average sales**

Based on sales momentum:

* **Trending** → strong positive growth
* **Stable** → minimal change
* **Fatiguing** → clear decline

This converts raw transactional data into a **supervised ML problem**.

---

## ⚙️ Feature Engineering

To help the model learn meaningful patterns, I engineered features such as:

* `sales_trend` – relative change in sales
* `sales_diff` – absolute sales change
* `sales_ratio` – recent vs past performance
* `momentum_score` – direction × magnitude
* `decline_flag` – explicit decline indicator

All features are **explainable** and tied to business intuition.

---

## 🤖 Models Used

* **Logistic Regression** – baseline, interpretable
* **Random Forest** – final model

### Evaluation Metric

* **Macro F1-score** (used to handle class imbalance correctly)

### Results

* Logistic Regression → Macro F1 ≈ **0.96**
* Random Forest → Macro F1 = **1.00**

> ⚠️ Note: High performance is expected because labels are derived from sales-based features. This is acknowledged as a limitation and discussed in the project.

---

## 🔍 Model Explainability

Feature importance analysis shows the model relies mainly on:

1. Sales trend
2. Sales difference
3. Momentum score

This aligns perfectly with how product fatigue was defined, making predictions easy to explain to non-technical stakeholders.

---

## 🚀 Deployment

The project is fully deployed:

### 🌐 Live API (FastAPI + Render)

* Predicts product status via REST API
* Auto-trains the model if not found
* Free cloud deployment

🔗 **API Docs:**
👉 [https://product-fatigue-api.onrender.com/docs](https://product-fatigue-api.onrender.com/docs)

---

### 🖥️ Live UI (Streamlit)

* Simple web interface
* Connected to the live API
* Allows anyone to test predictions interactively

🔗 **UI:**
👉 [https://fatigue-predictor.streamlit.app/](https://fatigue-predictor.streamlit.app/)

---

## 🗂️ Project Structure

```
product-fatigue-predictor/
│
├── app/                  # FastAPI backend
│   └── main.py
├── ui/                   # Streamlit UI
│   └── app.py
├── src/                  # ML pipeline
│   ├── label_generator.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 🔮 Future Improvements

Some realistic next steps:

* Predict **future fatigue**, not just classify past behavior
* Add seasonality awareness
* Include promotions or pricing changes
* Add automatic retraining schedules
* Build a richer dashboard for business users

---

## ✅ Final Thoughts

This project focuses on:

* Realistic data constraints
* Clear business logic
* End-to-end ML workflow
* Deployment and usability
* Explainability over hype

It reflects how machine learning is **actually built and shipped** in practice.

---

### 👤 Author

Built as a hands-on Machine Learning & Data Science portfolio project.

---


