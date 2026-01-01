import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
INPUT_PATH = "data/processed/preprocessed_data.csv"
OUTPUT_PATH = "data/processed/features.csv"


def load_data(path):
    return pd.read_csv(path)


def create_features(df):
    df = df.copy()

    df["sales_ratio"] = df["recent_avg_sales"] / (df["previous_avg_sales"] + 1)

    df["sales_diff"] = df["recent_avg_sales"] - df["previous_avg_sales"]

    df["momentum_score"] = df["sales_trend"] * df["recent_avg_sales"]

    df["decline_flag"] = (df["sales_trend"] < 0).astype(int)

    return df


def select_final_features(df):
    feature_cols = [
        "recent_avg_sales",
        "previous_avg_sales",
        "sales_trend",
        "sales_ratio",
        "sales_diff",
        "momentum_score",
        "decline_flag",
        "target"
    ]
    return df[feature_cols]


def main():
    df = load_data(INPUT_PATH)
    df = create_features(df)
    df = select_final_features(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
