from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# --- helpers ---
def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    # Replace non-breaking spaces, collapse multiple spaces, strip
    cols = (
        df.columns.astype(str)
          .str.replace("\u00a0", " ", regex=False)  # NBSP -> space
          .str.replace(r"\s+", " ", regex=True)    # collapse multi-spaces
          .str.strip()
    )
    df.columns = cols
    return df


def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = _normalize_headers(df)
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _normalize_headers(df)

    # Remove ID if present
    for col in ["customerID", "CustomerID", "customerId", "Customer Id", "CustomerID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # --- Target detection (prefer Churn Value if present, else Churn Label, else Churn) ---
    lower = {c: c.lower() for c in df.columns}
    col_churn_value = next((c for c, lc in lower.items() if lc == "churn value"), None)
    col_churn_label = next((c for c, lc in lower.items() if lc == "churn label"), None)
    col_churn_plain = next((c for c, lc in lower.items() if lc == "churn"), None)

    if col_churn_value is not None:
        df = df.rename(columns={col_churn_value: "Churn"})
        df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(0).astype(int)
    elif col_churn_label is not None:
        df = df.rename(columns={col_churn_label: "Churn"})
        df["Churn"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    elif col_churn_plain is not None:
        if col_churn_plain != "Churn":
            df = df.rename(columns={col_churn_plain: "Churn"})
        if df["Churn"].dtype == object:
            df["Churn"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
        else:
            df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(0).astype(int)
    else:
        # Give a helpful error listing the columns we actually see
        raise ValueError(
            "Could not find a churn target. Looked for 'Churn Value', 'Churn Label', or 'Churn'. "
            f"Columns found: {list(df.columns)}"
        )

    # Total Charges -> numeric (handles 'Total Charges' and 'TotalCharges')
    for tc in ["Total Charges", "TotalCharges"]:
        if tc in df.columns:
            df[tc] = pd.to_numeric(df[tc], errors="coerce")

    if "Total Charges" in df.columns:
        df = df.loc[~df["Total Charges"].isna()].copy()
    elif "TotalCharges" in df.columns:
        df = df.loc[~df["TotalCharges"].isna()].copy()

    # Convert string/object columns (except target) to category
    for c in df.columns:
        if c != "Churn" and df[c].dtype == object:
            df[c] = df[c].astype("category")

    return df


def split_Xy(df: pd.DataFrame, target: str = "Churn") -> Tuple[pd.DataFrame, np.ndarray]:
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found after preparation. Columns: {list(df.columns)}")
    X = df.drop(columns=[target])
    y = df[target].to_numpy().astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features: List[str] = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features: List[str] = [
        c for c in X.columns
        if pd.api.types.is_categorical_dtype(X[c]) or X[c].dtype == object
    ]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features

