from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def load_data(path: str) -> pd.DataFrame:
    """Load Telco churn CSV/XLSX exactly like the R script expects."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    return df

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror the R preprocessing:
      - drop customerID
      - TotalCharges -> numeric and drop NA rows
      - Churn (Yes/No) -> 1/0
      - object columns -> category (factor)
    """
    df = df.copy()

    # Drop ID if present
    for col in ["customerID", "CustomerID", "customerId", "Customer Id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # TotalCharges -> numeric + drop NAs
    for tc in ["TotalCharges", "Total Charges"]:
        if tc in df.columns:
            df[tc] = pd.to_numeric(df[tc].astype(str).str.strip(), errors="coerce")
            df = df.loc[~df[tc].isna()].copy()
            break

    # Encode Churn Yes/No -> 1/0
    if "Churn" in df.columns:
        df["Churn"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    elif "Churn Label" in df.columns:
        df["Churn"] = (df["Churn Label"].astype(str).str.strip().str.lower() == "yes").astype(int)
    else:
        raise ValueError("Could not find 'Churn' (or 'Churn Label') column.")

    # Object -> category (R factors)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")

    return df

def split_Xy(df: pd.DataFrame, target: str = "Churn") -> Tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[target])
    y = df[target].to_numpy().astype(int)
    return X, y
PY

