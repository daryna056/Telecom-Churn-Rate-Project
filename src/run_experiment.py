<<<<<<< HEAD
# ======================
# Telco Churn Prediction — Python mirror of the R project
# ======================

import os
import io
import json
import pandas as pd
import numpy as np
=======
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
>>>>>>> a147d3fa1050dae34da7e04e8c802726d7eed3ca
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
<<<<<<< HEAD
    confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score,
    f1_score, accuracy_score, cohen_kappa_score
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp

import statsmodels.api as sm  # ✅ Needed for GLM families
import warnings
warnings.filterwarnings("ignore")

# ------------------------
# Paths & output helpers
# ------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Telco_customer_churn.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_txt(text, filename):
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        f.write(text if isinstance(text, str) else json.dumps(text, indent=2))

def save_fig(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()

# ------------------------
# 2. Load & Prepare Data
# ------------------------
print("📂 Loading data:", DATA_PATH)
df = pd.read_excel(DATA_PATH, sheet_name=0, header=0)

# --- Normalize column names (strip spaces, remove inner spaces) ---
df.columns = (
    df.columns.astype(str)
    .str.replace("\uFEFF", "", regex=False)
    .str.strip()
    .str.replace(r"\s+", "", regex=True)
)

# --- Create 'Churn' as in R dataset (Yes/No -> 1/0) ---
if "Churn" not in df.columns:
    if "ChurnLabel" in df.columns:
        churn_str = df["ChurnLabel"].astype(str).str.strip().str.lower()
        df["Churn"] = np.where(churn_str.eq("yes"), 1, 0)
    elif "ChurnValue" in df.columns:
        df["Churn"] = pd.to_numeric(df["ChurnValue"], errors="coerce").fillna(0).astype(int)
    else:
        raise SystemExit("❌ Could not find Churn/ChurnLabel/ChurnValue to construct target 'Churn'.")

# --- Drop leakage columns that R didn’t include ---
for leak_col in ["ChurnLabel", "ChurnValue", "ChurnScore", "ChurnReason"]:
    if leak_col in df.columns:
        df.drop(columns=[leak_col], inplace=True)

# --- Drop ID column (CustomerID) ---
id_cols = [c for c in df.columns if c.lower() == "customerid"]
if id_cols:
    df.drop(columns=id_cols, inplace=True)

# --- Coerce TotalCharges, drop NAs (exactly like R) ---
if "TotalCharges" not in df.columns:
    raise SystemExit("❌ Expected 'TotalCharges' column after normalization.")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).copy()

# --- Clean up string cells and make factors (categoricals) ---
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.strip()
df["Churn"] = df["Churn"].astype(int)

char_cols = df.select_dtypes(include=["object"]).columns.tolist()
for c in char_cols:
    df[c] = df[c].astype("category")

# ------------------------
# 3. Train-Test Split
# ------------------------
np.random.seed(123)
X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

cat_features = X.select_dtypes(include=["category", "object"]).columns.tolist()
num_features = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features),
    ],
    remainder="drop"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

print("Train Churn Distribution:")
print(y_train.value_counts().sort_index())
print("Test Churn Distribution:")
print(y_test.value_counts().sort_index())

save_txt(
    "Train distribution:\n" + str(y_train.value_counts().sort_index()) +
    "\n\nTest distribution:\n" + str(y_test.value_counts().sort_index()),
    "01_split_distributions.txt"
)

# ------------------------
# 4. Evaluation Function
# ------------------------
def evaluate_model(actual, predicted_prob, predicted_class, model_name):
    cm = confusion_matrix(actual, predicted_class, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(actual, predicted_class),
        "Precision": precision_score(actual, predicted_class, zero_division=0),
        "Recall": recall_score(actual, predicted_class, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "F1_Score": f1_score(actual, predicted_class, zero_division=0),
        "AUC_ROC": roc_auc_score(actual, predicted_prob),
    }
    return metrics, cm

# ------------------------
# 5. Logistic Regression (GLM)
# ------------------------
train_glm = pd.concat([X_train.copy(), y_train.rename("Churn")], axis=1)
for c in cat_features:
    train_glm[c] = train_glm[c].astype(str)

formula = "Churn ~ " + " + ".join([f"C({c})" for c in cat_features] + num_features)
logit_model_sm = sm.GLM.from_formula(formula, data=train_glm, family=sm.families.Binomial()).fit()

glm_buffer = io.StringIO()
glm_buffer.write("Logistic Model Summary (statsmodels GLM on TRAIN):\n")
glm_buffer.write(str(logit_model_sm.summary()) + "\n\n")
glm_buffer.write("Odds Ratios:\n")
glm_buffer.write(str(np.exp(logit_model_sm.params)) + "\n\n")
glm_buffer.write("LogLik, AIC, BIC:\n")
bic = getattr(logit_model_sm, "bic", np.nan)
glm_buffer.write(str((logit_model_sm.llf, logit_model_sm.aic, bic)) + "\n")
save_txt(glm_buffer.getvalue(), "02_logistic_glm_summary.txt")

log_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
])
log_model.fit(X_train, y_train)

log_pred_test = log_model.predict_proba(X_test)[:, 1]
log_class_test = (log_pred_test > 0.5).astype(int)
log_pred_train = log_model.predict_proba(X_train)[:, 1]
log_class_train = (log_pred_train > 0.5).astype(int)

# ------------------------
# 6. Decision Trees (Gini & Entropy)
# ------------------------
tree_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", DecisionTreeClassifier(criterion="gini", random_state=123))
])
tree_model.fit(X_train, y_train)

tree_model2 = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", DecisionTreeClassifier(criterion="entropy", random_state=123))
])
tree_model2.fit(X_train, y_train)

tree_pred_test = tree_model.predict_proba(X_test)[:, 1]
tree_class_test = (tree_pred_test > 0.5).astype(int)
tree_pred_train = tree_model.predict_proba(X_train)[:, 1]
tree_class_train = (tree_pred_train > 0.5).astype(int)

# ------------------------
# 7. Random Forest
# ------------------------
rf_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=500, random_state=123, n_jobs=-1, criterion="gini"
    ))
])
rf_model.fit(X_train, y_train)

rf_pred_test = rf_model.predict_proba(X_test)[:, 1]
rf_class_test = (rf_pred_test > 0.5).astype(int)
rf_pred_train = rf_model.predict_proba(X_train)[:, 1]
rf_class_train = (rf_pred_train > 0.5).astype(int)

# ------------------------
# 8. Gradient Boosting (GBM)
# ------------------------
gbm_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=3,
        min_samples_leaf=10,
        random_state=123
    ))
])
gbm_model.fit(X_train, y_train)

gbm_pred_test = gbm_model.predict_proba(X_test)[:, 1]
gbm_class_test = (gbm_pred_test > 0.5).astype(int)
gbm_pred_train = gbm_model.predict_proba(X_train)[:, 1]
gbm_class_train = (gbm_pred_train > 0.5).astype(int)

# ------------------------
# 9–10. Evaluate All Models
# ------------------------
metrics_test = []
metrics_train = []
cms = {}

for name, (pred_test, cls_test, pred_train, cls_train) in {
    "Logistic Regression": (log_pred_test, log_class_test, log_pred_train, log_class_train),
    "Decision Tree":       (tree_pred_test, tree_class_test, tree_pred_train, tree_class_train),
    "Random Forest":       (rf_pred_test,  rf_class_test,  rf_pred_train,  rf_class_train),
    "Gradient Boosting":   (gbm_pred_test, gbm_class_test, gbm_pred_train, gbm_class_train),
}.items():
    mtest, cmtest = evaluate_model(y_test, pred_test, cls_test, name)
    mtrain, cmtrain = evaluate_model(y_train, pred_train, cls_train, name)
    metrics_test.append(mtest)
    metrics_train.append(mtrain)
    cms[name] = {"test_confusion_matrix": cmtest.tolist(),
                 "train_confusion_matrix": cmtrain.tolist()}

df_test = pd.DataFrame(metrics_test)
df_train = pd.DataFrame(metrics_train)
df_test.to_excel(os.path.join(OUTPUT_DIR, "03_metrics_test.xlsx"), index=False)
df_train.to_excel(os.path.join(OUTPUT_DIR, "04_metrics_train.xlsx"), index=False)
save_txt(cms, "05_confusion_matrices.json")

print("\n==== Evaluation on Test Set ====")
print(df_test)
print("\n==== Evaluation on Train Set ====")
print(df_train)

# ------------------------
# 11. ROC Curves
# ------------------------
plt.figure(figsize=(8,6))
for label, probs in [
    ("Logistic", log_pred_test),
    ("Decision Tree", tree_pred_test),
    ("Random Forest", rf_pred_test),
    ("GBM", gbm_pred_test)
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
plt.plot([0,1],[0,1], linestyle="--")
plt.title("ROC Curves - Test Set")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
save_fig("06_roc_curves_test.png")

print("\n✅ Finished. All artifacts saved to:", OUTPUT_DIR)
=======
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

>>>>>>> a147d3fa1050dae34da7e04e8c802726d7eed3ca
