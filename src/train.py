import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

DATA_PATH = "data/processed/features.csv"
MODEL_OUTPUT_PATH = "models/fatigue_model.pkl"
RANDOM_STATE = 42


def load_data(path):
    return pd.read_csv(path)


def split_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)

    print(f"\n{name} Results")
    print("-" * 40)
    print("Macro F1:", f1_score(y_test, preds, average="macro"))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))


def main():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Save best model (Random Forest)
    joblib.dump(rf_model, MODEL_OUTPUT_PATH)
    print(f"\nBest model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
