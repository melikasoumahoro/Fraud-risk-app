from __future__ import annotations
import re
import pandas as pd

DEFAULT_DROP_REGEX = re.compile(
    r"(name|phone|email|address|addr|account|customer|id|meter|passport|ssn)",
    re.IGNORECASE,
)

def split_X_y(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)[:30]} ...")

    y = df[target_col].astype(int) if df[target_col].dropna().isin([0, 1]).all() else df[target_col]
    X = df.drop(columns=[target_col])

    # Drop obviously identifying columns
    drop_cols = [c for c in X.columns if DEFAULT_DROP_REGEX.search(str(c))]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    return X, y

def basic_clean(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # Convert booleans to int
    for c in X.select_dtypes(include=["bool"]).columns:
        X[c] = X[c].astype(int)

    # Strip whitespace from object columns
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype(str).str.strip()

    return X