from pathlib import Path
import json

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


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


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLUMNS]


def main() -> None:
    train_parquet_dir = Path("data/processed/train_parquet_chunks2")
    model_path = Path("models/model2.joblib")
    feature_columns_path = Path("models/feature_columns2.json")
    train_metrics_path = Path("artifacts/metrics/train_metrics2.json")

    print(f"Loading parquet chunks from: {train_parquet_dir}")
    train_df = load_parquet_dir(train_parquet_dir)

    if "label" not in train_df.columns:
        raise ValueError("label column not found in train parquet data")

    feature_columns = get_feature_columns(train_df)

    X_train = train_df[feature_columns].copy()
    y_train = train_df["label"].astype("int8").copy()

    print(f"Training rows: {len(train_df)}")
    print(f"Number of features: {len(feature_columns)}")

    # Old:
    # model = RandomForestClassifier(
    #     bootstrap=True,
    #     criterion="gini",
    #     max_depth=14,
    #     max_features="sqrt",
    #     max_leaf_nodes=None,
    #     min_impurity_decrease=0.0,
    #     min_samples_leaf=28,
    #     min_samples_split=111,
    #     min_weight_fraction_leaf=0.0,
    #     n_estimators=121,
    #     n_jobs=-1,
    #     oob_score=False,
    #     random_state=25,
    #     verbose=0,
    #     warm_start=False,
    # )

    model = LGBMClassifier(
        boosting_type="gbdt",
        num_leaves=150,
        max_depth=15,
        learning_rate=0.05,
        n_estimators=1000,
        objective="binary",
        class_weight="balanced",  # Help with recall for minorities if any, though here it might help shift threshold
        random_state=25,
        n_jobs=-1,
        verbose=-1,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating on training data...")
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred)

    print(f"Train Accuracy: {train_acc:.6f}")
    print(f"Train F1: {train_f1:.6f}")
    print(classification_report(y_train, train_pred))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    with open(feature_columns_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    with open(train_metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_accuracy": train_acc,
                "train_f1": train_f1,
                "n_train_rows": len(train_df),
                "n_features": len(feature_columns),
            },
            f,
            indent=2,
        )

    print("Done.")


if __name__ == "__main__":
    main()