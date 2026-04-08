from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


DROP_COLUMNS = {"source_node", "destination_node", "label"}


def load_parquet_dir(parquet_dir: str | Path) -> pd.DataFrame:
    parquet_dir = Path(parquet_dir)
    files = sorted(parquet_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in: {parquet_dir}")

    dfs = []
    for f in files:
        print(f"Loading {f}")
        dfs.append(pd.read_parquet(f))

    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    test_parquet_dir = Path("data/processed/test_parquet_chunks2")
    model_path = Path("models/model2.joblib")
    feature_columns_path = Path("models/feature_columns2.json")
    metrics_path = Path("artifacts/metrics/test_metrics2.json")
    feature_importance_path = Path("models/feature_importance2.csv")

    print(f"Loading test parquet chunks from: {test_parquet_dir}")
    test_df = load_parquet_dir(test_parquet_dir)

    if "label" not in test_df.columns:
        raise ValueError("label column not found in test parquet data")

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading feature columns from: {feature_columns_path}")
    with open(feature_columns_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    missing_features = [col for col in feature_columns if col not in test_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in test data: {missing_features}")

    X_test = test_df[feature_columns].copy()
    y_test = test_df["label"].astype("int8").copy()

    print("Predicting on test data...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Test Accuracy: {test_acc:.6f}")
    print(f"Test F1: {test_f1:.6f}")
    print(f"Test ROC-AUC: {test_auc:.6f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred))

    metrics = {
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "test_roc_auc": test_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_test_rows": len(test_df),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved test metrics to: {metrics_path}")

    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        fi.to_csv(feature_importance_path, index=False)
        print(f"Saved feature importance to: {feature_importance_path}")

        print("\nTop 15 important features:")
        print(fi.head(15))

    print("Evaluation complete.")


if __name__ == "__main__":
    main()