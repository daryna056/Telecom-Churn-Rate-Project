
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, confusion_matrix

# Flexible imports (module or script)
try:
    from .data_processing import load_data, prepare_dataframe, split_Xy, build_preprocessor
    from .models import get_models
    from .evaluate import (
        evaluate_all, plot_roc, plot_confusion, lift_curve, plot_lift_curves,
        ks_curve_arrays, plot_ks_curves, plot_feature_importance
    )
except Exception:
    import sys
    base = Path(__file__).resolve().parents[0]
    sys.path.append(str(base))
    from data_processing import load_data, prepare_dataframe, split_Xy, build_preprocessor
    from models import get_models
    from evaluate import (
        evaluate_all, plot_roc, plot_confusion, lift_curve, plot_lift_curves,
        ks_curve_arrays, plot_ks_curves, plot_feature_importance
    )

def main(args):
    root = Path(__file__).resolve().parents[1]
    data_path = root / "Telco_customer_churn.xlsx"
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True, parents=True)

    # Load & prep
    df = load_data(str(data_path))
    df = prepare_dataframe(df)
    X, y = split_Xy(df, target="Churn")

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    models = get_models(random_state=args.seed)

    # Containers
    metrics_test = []
    metrics_train = []
    roc_curves = {}
    lift_curves = {}
    ks_curves = {}

    # Fit/evaluate
    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        # Train predictions
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            proba_tr = pipe.predict_proba(X_train)[:, 1]
            proba_te = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe.named_steps["model"], "decision_function"):
            proba_tr = pipe.decision_function(X_train)
            proba_te = pipe.decision_function(X_test)
        else:
            proba_tr = None
            proba_te = None

        pred_tr = pipe.predict(X_train)
        pred_te = pipe.predict(X_test)

        # Metrics
        metrics_train.append(evaluate_all(y_train, proba_tr, pred_tr, name))
        metrics_test.append(evaluate_all(y_test, proba_te, pred_te, name))

        # ROC
        if proba_te is not None:
            fpr, tpr, _ = roc_curve(y_test, proba_te)
            roc_curves[name] = (fpr, tpr)

        # Lift
        if proba_te is not None:
            x, y_lift = lift_curve(y_test, proba_te)
            lift_curves[name] = (x, y_lift)

        # KS curves
        if proba_te is not None:
            cum_pos, cum_neg, ks_val = ks_curve_arrays(y_test, proba_te)
            ks_curves[name] = (cum_pos, cum_neg, ks_val)

        # Confusion heatmap
        cm = confusion_matrix(y_test, pred_te)
        plot_confusion(cm, ["No", "Yes"], f"Confusion Matrix - {name} (Test)", str(outputs / f"confusion_matrix_{name.replace(' ', '_')}.png"))

        # Feature importance (if available)
        model_step = pipe.named_steps["model"]
        importances = getattr(model_step, "feature_importances_", None)
        if importances is not None:
            # Get feature names from preprocessor
            ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
            num_features = pipe.named_steps["prep"].transformers_[0][2]
            cat_features = list(ohe.get_feature_names_out(pipe.named_steps["prep"].transformers_[1][2]))
            feature_names = list(num_features) + cat_features
            plot_feature_importance(importances, feature_names, f"Top Features - {name}", str(outputs / f"feature_importance_{name.replace(' ', '_')}.png"))

    # Save metrics
    df_test = pd.DataFrame(metrics_test)
    df_train = pd.DataFrame(metrics_train)
    df_test.to_csv(outputs / "metrics_test.csv", index=False)
    df_train.to_csv(outputs / "metrics_train.csv", index=False)

    # Save curves
    if roc_curves:
        plot_roc(roc_curves, str(outputs / "roc_curves.png"))
    if lift_curves:
        plot_lift_curves(lift_curves, str(outputs / "lift_curves.png"))
    if ks_curves:
        plot_ks_curves(ks_curves, str(outputs / "ks_curves.png"))

    print("Done. Metrics and plots saved in:", outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
