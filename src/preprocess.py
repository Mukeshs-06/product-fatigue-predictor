import pandas as pd

INPUT_PATH = "data/processed/labeled_products.csv"
OUTPUT_PATH = "data/processed/preprocessed_data.csv"

TARGET_MAP = {
    "Fatiguing": 0,
    "Stable": 1,
    "Trending": 2
}


def load_data(path):
    df = pd.read_csv(path)
    return df


def basic_cleaning(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=["product_status"])
    return df


def encode_target(df):
    df["target"] = df["product_status"].map(TARGET_MAP)
    return df


def select_columns(df):
    keep_cols = [
        "SKU",
        "recent_avg_sales",
        "previous_avg_sales",
        "sales_trend",
        "target"
    ]
    return df[keep_cols]


def main():
    df = load_data(INPUT_PATH)
    df = basic_cleaning(df)
    df = encode_target(df)
    df = select_columns(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessed data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
