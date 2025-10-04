cat > src/run_experiment.py << 'PY'
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, cohen_kappa_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import ks_2samp

from data_processing import load_data, prepare_dataframe, split_Xy

# ---------- helpers ----------
def compute_metrics(y_true, y_prob, y_pred, model_name, dataset):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)  # Pos Pred Value
    rec = recall_score(y_true, y_pred)                       # Sensitivity
    spec = tn / (tn + fp) if (tn + fp) else 0.0              # Specificity
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    kappa = cohen_kappa_score(y_true, y_pred)
    return {
        "Model": model_name,
        "Dataset": dataset,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "F1_Score": f1,
        "AUC_ROC": auc,
        "Cohen_Kappa": kappa,
    }, cm

def lift_curve_points(y_true, y_prob):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    df["cum_pos"] = np.cumsum(df["y"])
    total_pos = df["y"].sum()
    df["perc_data"] = np.arange(1, len(df) + 1) / len(df)
    df["perc_target"] = df["cum_pos"] / total_pos if total_pos > 0 else 0.0
    df["lift"] = df["perc_target"] / df["perc_data"]
    return df["perc_data"].values, df["lift"].values

def ks_statistic(y_true, y_prob):
    return ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0])

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="WA_Fn-UseC_-Telco-Customer-Churn.csv",
                        help="Path to IBM Telco CSV (or Excel).")
    args = parser.parse_args()

    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)

    # 1) Load + prep (mirror R)
    df = load_data(args.data)
    df = prepare_dataframe(df)
    X, y = split_Xy(df, target="Churn")

    # 2) Stratified 80/20 split with seed=123 (like caret::createDataPartition)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    # 3) One-hot (R factors -> design matrix). No scaling (R didn’t scale).
    X_tr_mat = pd.get_dummies(X_tr, drop_first=False)
    X_te_mat = pd.get_dummies(X_te, drop_first=False)
    X_te_mat = X_te_mat.reindex(columns=X_tr_mat.columns, fill_value=0)

    # 4) Models aligned to R choices/params
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "Decision Tree (Gini)": DecisionTreeClassifier(criterion="gini", random_state=123),
        "Decision Tree (Entropy)": DecisionTreeClassifier(criterion="entropy", random_state=123),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, criterion="gini", random_state=123
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.01, max_depth=3,
            min_samples_leaf=10, random_state=123
        ),
    }

    metrics_train, metrics_test = [], []
    roc_store, lift_store = {}, {}
    ks_rows, importances = [], {}

    for name, clf in models.items():
        clf.fit(X_tr_mat, y_tr)
        y_prob_tr = clf.predict_proba(X_tr_mat)[:, 1]
        y_prob_te = clf.predict_proba(X_te_mat)[:, 1]
        y_pred_tr = (y_prob_tr > 0.5).astype(int)
        y_pred_te = (y_prob_te > 0.5).astype(int)

        mtr, cm_tr = compute_metrics(y_tr, y_prob_tr, y_pred_tr, name, "Train")
        mte, cm_te = compute_metrics(y_te, y_prob_te, y_pred_te, name, "Test")
        metrics_train.append(mtr); metrics_test.append(mte)

        print(f"\n{name} - Confusion Matrix (Test):\n{cm_te}")

        fpr, tpr, _ = roc_curve(y_te, y_prob_te)
        roc_store[name] = (fpr, tpr, roc_auc_score(y_te, y_prob_te))

        px, py = lift_curve_points(y_te, y_prob_te)
        lift_store[name] = (px, py)

        ks = ks_statistic(y_te, y_prob_te)
        ks_rows.append({"Model": name, "KS_Statistic": ks.statistic, "KS_pvalue": ks.pvalue})

        if hasattr(clf, "feature_importances_"):
            importances[name] = pd.Series(clf.feature_importances_, index=X_tr_mat.columns) \
                                    .sort_values(ascending=False)

    # Save metrics
    df_train = pd.DataFrame(metrics_train)
    df_test = pd.DataFrame(metrics_test)
    df_train.to_csv(os.path.join(outdir, "metrics_train.csv"), index=False)
    df_test.to_csv(os.path.join(outdir, "metrics_test.csv"), index=False)
    pd.DataFrame(ks_rows).to_csv(os.path.join(outdir, "metrics_ks.csv"), index=False)

    print("\n==== Evaluation on Train Set ====")
    print(df_train.round(3))
    print("\n==== Evaluation on Test Set ====")
    print(df_test.round(3))

    # ROC (all)
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_store.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Test Set"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curves.png")); plt.close()

    # Lift (all)
    plt.figure(figsize=(7, 6))
    for name, (px, py) in lift_store.items():
        plt.plot(px, py, label=name)
    plt.axhline(1.0, color="black", linestyle="--")
    plt.xlabel("Proportion of Test Set (sorted by score)")
    plt.ylabel("Lift")
    plt.title("Lift Curves - Test Set"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lift_curves.png")); plt.close()

    # KS (per model)
    for name in roc_store.keys():
        # recompute sorted CDFs for curve
        y_prob_te = models[name].predict_proba(X_te_mat)[:, 1]
        dfks = pd.DataFrame({"y": y_te, "p": y_prob_te}).sort_values("p", ascending=False)
        dfks["cum_pos"] = np.cumsum(dfks["y"]) / dfks["y"].sum()
        dfks["cum_neg"] = np.cumsum(1 - dfks["y"]) / (len(dfks) - dfks["y"].sum())
        ks_stat = np.max(np.abs(dfks["cum_pos"] - dfks["cum_neg"]))

        plt.figure(figsize=(7, 6))
        plt.plot(dfks["cum_pos"].values, label="Positive (1)")
        plt.plot(dfks["cum_neg"].values, label="Negative (0)")
        plt.title(f"Kolmogorov–Smirnov Curve - {name} (KS={ks_stat:.3f})")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"ks_{name.replace(' ', '_')}.png")); plt.close()

    # Feature importance (RF & GBM)
    for model_name in ["Random Forest", "Gradient Boosting"]:
        if model_name in importances:
            imp = importances[model_name].head(10)
            plt.figure(figsize=(8, 6))
            plt.barh(imp.index[::-1], imp.values[::-1])
            plt.title(f"Top 10 Important Features ({model_name})")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"feature_importance_{model_name.replace(' ', '_')}.png"))
            plt.close()

    print(f"\nDone. Metrics and plots saved in: {os.path.abspath(outdir)}")

if __name__ == "__main__":
    main()
PY

