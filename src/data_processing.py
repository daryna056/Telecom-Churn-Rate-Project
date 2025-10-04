
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove ID if present
    for col in ["customerID", "CustomerID", "customerId", "Customer Id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # TotalCharges to numeric (as in R)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.loc[~df["TotalCharges"].isna()].copy()

    # Encode Churn Yes/No -> 1/0
    if "Churn" in df.columns:
        df["Churn"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Convert object columns to category
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("category")

    return df

def split_Xy(df: pd.DataFrame, target: str = "Churn") -> Tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[target])
    y = df[target].to_numpy().astype(int)
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Identify numeric and categorical columns
    numeric_features: List[str] = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features: List[str] = [c for c in X.columns if pd.api.types.is_categorical_dtype(X[c]) or X[c].dtype == object]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features
