

# 📦 Product Fatigue Predictor

### Predict when a product is **Trending**, **Stable**, or **Losing Momentum**

---

## 👋 What is this project?

In real businesses, one of the hardest questions is:

> *“Is this product still performing well, or is it slowly dying?”*

This project answers that question using **sales data**.

I built an **end-to-end machine learning system** that analyzes historical sales behavior and predicts whether a product is:

* 🔥 **Trending** – demand is increasing
* 😐 **Stable** – performance is steady
* 💤 **Fatiguing** – sales momentum is declining

The entire pipeline goes from **raw data → ML model → deployed API**.

---

## 🧠 Why this problem matters

Most datasets **do not come with labels** like *“this product is fatiguing”*.
So instead of relying on pre-labeled data, this project:

* **Creates labels using business logic**
* Mimics how real teams reason about product performance
* Focuses on **interpretability**, not black-box predictions

This makes the project realistic and interview-ready.

---

## 📂 Dataset

* Transaction-level ecommerce sales data
* Each row represents an order
* Key columns used:

  * `SKU` – product identifier
  * `Date` – order date
  * `Qty` – units sold
  * `Amount` – revenue

No ratings or reviews were used — fatigue is inferred **purely from sales behavior**, which is common in real businesses.

---

## 🏷️ How labels are created (core idea)

Since fatigue labels don’t exist naturally, I **engineered them**.

For each product (`SKU`):

1. Aggregate daily sales
2. Build a continuous time series (missing days → zero sales)
3. Compare:

   * **Recent 30-day average sales**
   * **Previous 30-day average sales**

Based on sales momentum:

* **Trending** → strong positive change
* **Stable** → minimal change
* **Fatiguing** → clear decline

This turns an unlabeled dataset into a **supervised ML problem**.

---

## ⚙️ Feature Engineering

To help the model learn meaningful patterns, I engineered features such as:

* `sales_trend` – relative change in demand
* `sales_diff` – absolute sales change
* `sales_ratio` – recent vs past performance
* `momentum_score` – direction × volume
* `decline_flag` – explicit decline indicator

All features are **explainable** and grounded in business logic.

---

## 🤖 Models Used

Two models were trained and compared:

* **Logistic Regression** (baseline, interpretable)
* **Random Forest** (final model)

### Evaluation Metric

* **Macro F1-score** (chosen to handle class imbalance properly)

### Results

* Logistic Regression → Macro F1 ≈ **0.96**
* Random Forest → Macro F1 = **1.00**

> ⚠️ Note: High performance is expected because labels are derived from sales-based features. This makes the task deterministic and transparent, which is acknowledged as a limitation.

---

## 🔍 Model Explainability

Feature importance analysis shows the model relies mainly on:

1. Sales trend
2. Sales difference
3. Momentum score

This aligns directly with how fatigue was defined, making predictions easy to explain to non-technical stakeholders.

---

## 🚀 Deployment (FastAPI)

The trained model is deployed as a REST API using **FastAPI**.

### Example request

```json
{
  "recent_avg_sales": 0.3,
  "previous_avg_sales": 0.6,
  "sales_trend": -0.4,
  "sales_ratio": 0.5,
  "sales_diff": -0.3,
  "momentum_score": -0.12,
  "decline_flag": 1
}
```

### Example response

```json
{
  "prediction": "Fatiguing",
  "confidence_note": "Product is losing sales momentum"
}
```

---

## 🗂️ Project Structure

```
product-fatigue-predictor/
│
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── label_generator.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   └── fatigue_model.pkl
├── app/
│   └── main.py
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

If this were extended further, I would:

* Predict **future fatigue**, not just classify past behavior
* Add seasonality awareness
* Include promotions or pricing changes
* Build a dashboard for business users

---

## ✅ Final Note

This project focuses on:

* Realistic data constraints
* Clear business logic
* End-to-end ML workflow
* Explainability over hype

It reflects how machine learning is actually used in practice.

---
