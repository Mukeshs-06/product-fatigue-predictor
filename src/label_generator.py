import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/raw/products.csv"
OUTPUT_PATH = "data/processed/labeled_products.csv"

RECENT_DAYS = 30
THRESHOLD = 0.10


def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df


def aggregate_daily_sales(df):
    daily = (
        df.groupby(['SKU', 'Date'])
          .agg(daily_qty=('Qty', 'sum'))
          .reset_index()
    )
    return daily


def make_continuous_series(daily_df):
    all_products = []

    for sku, sku_df in daily_df.groupby('SKU'):
        sku_df = sku_df.sort_values('Date')

        date_range = pd.date_range(
            start=sku_df['Date'].min(),
            end=sku_df['Date'].max(),
            freq='D'
        )

        sku_df = (
            sku_df
            .set_index('Date')
            .reindex(date_range, fill_value=0)
            .rename_axis('Date')
            .reset_index()
        )

        sku_df['SKU'] = sku
        all_products.append(sku_df)

    return pd.concat(all_products, ignore_index=True)


def generate_labels(df):
    labels = []

    for sku, sku_df in df.groupby('SKU'):
        if len(sku_df) < RECENT_DAYS * 2:
            continue  

        recent = sku_df.tail(RECENT_DAYS)['daily_qty'].mean()
        previous = sku_df.iloc[-(RECENT_DAYS*2):-RECENT_DAYS]['daily_qty'].mean()

        trend = (recent - previous) / (previous + 1)

        if trend >= THRESHOLD:
            label = "Trending"
        elif trend <= -THRESHOLD:
            label = "Fatiguing"
        else:
            label = "Stable"

        labels.append({
            "SKU": sku,
            "recent_avg_sales": recent,
            "previous_avg_sales": previous,
            "sales_trend": trend,
            "product_status": label
        })

    return pd.DataFrame(labels)


def main():
    df = load_data(RAW_DATA_PATH)
    daily_sales = aggregate_daily_sales(df)
    continuous_sales = make_continuous_series(daily_sales)
    labeled_df = generate_labels(continuous_sales)

    labeled_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved labeled data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
