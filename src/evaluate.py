import pandas as pd
import joblib
import matplotlib.pyplot as plt

MODEL_PATH = "models/fatigue_model.pkl"
DATA_PATH = "data/processed/features.csv"


def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)

    importances = model.feature_importances_
    feature_names = X.columns

    importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })
        .sort_values(by="importance", ascending=False)
    )

    print(importance_df)

    # Plot
    plt.figure(figsize=(8,5))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance — Product Fatigue Model")
    plt.show()


if __name__ == "__main__":
    main()
