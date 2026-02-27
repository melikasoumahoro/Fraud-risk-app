from __future__ import annotations
import argparse
import json
from os import pipe
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from preprocess import split_X_y, basic_clean

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    X = basic_clean(X)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=None,
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
    return pipe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/fraud.csv")
    parser.add_argument("--target", required=True, help="Name of label column (0/1 fraud)")
    parser.add_argument("--out", default="model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    X, y = split_X_y(df, args.target)
    X = basic_clean(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.dropna().isin([0, 1]).all() else None
    )

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    # --- Feature importance (for tree models) ---
    importances = None
    feature_names = None

    try:
        model = pipe.named_steps["model"]
        pre = pipe.named_steps["preprocess"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = pre.get_feature_names_out().tolist()
    except Exception:
        pass

    # Evaluate
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = None
    else:
        proba = None
        auc = None

    preds = pipe.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, preds))

    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")

    proba = pipe.predict_proba(X_test)[:, 1]

    for threshold in [0.5, 0.4, 0.3, 0.2]:
        preds_thresh = (proba >= threshold).astype(int)
        print(f"\nThreshold: {threshold}")
        print(classification_report(y_test, preds_thresh))

    # Save model + metadata
    artifacts = {
    "pipeline": pipe,
    "target_col": args.target,
    "columns_seen": X.columns.tolist(),
    "feature_names": feature_names,
    "feature_importances": importances,
    }
    joblib.dump(artifacts, args.out)

    with open("metrics.json", "w") as f:
        json.dump(
            {
                "roc_auc": auc,
                "fraud_rate": float(pd.Series(y).mean()),
                "recommended_threshold": 0.30
         },
         f,
         indent=2
     )

    print(f"Saved model to: {args.out}")

if __name__ == "__main__":
    main()