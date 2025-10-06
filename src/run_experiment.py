# ======================
# Telco Churn Prediction — Python version of your R workflow
# ======================

import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# --- OneHotEncoder compatibility (sklearn >=1.2 uses sparse_output) ---
from sklearn.preprocessing import OneHotEncoder
def make_ohe():
    import sklearn
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 2):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ------------------------
# 0) Paths & helpers
# ------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
CANDIDATES = [
    os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    os.path.join(BASE_DIR, "WA_Fn-UseC_-Telc-Customer-Churn.csv"),
]
DATA_PATH  = next((p for p in CANDIDATES if os.path.exists(p)), CANDIDATES[0])
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
# 1) Load & prepare data (matches your R preprocessing)
# ------------------------
print("📂 Loading data:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Remove ID, convert TotalCharges to numeric, drop NA rows
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).copy()

# Encode Churn: Yes→1, No→0
df["Churn"] = np.where(df["Churn"].astype(str).str.strip().str.lower() == "yes", 1, 0)

# Convert object columns to category (R factors analogue)
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype("category")

# ------------------------
# 2) Train/Test split (seed=123; 80/20; stratified like caret)
# ------------------------
np.random.seed(123)
X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

cat_features = X.select_dtypes(include=["category", "object"]).columns.tolist()
num_features = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", make_ohe(), cat_features),
        ("num", "passthrough", num_features),
    ],
    remainder="drop"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

print("Train Churn Distribution:\n", y_train.value_counts().sort_index())
print("Test Churn Distribution:\n", y_test.value_counts().sort_index())
save_txt(
    f"Train:\n{y_train.value_counts().sort_index()}\n\nTest:\n{y_test.value_counts().sort_index()}",
    "01_split_distributions.txt"
)

# ------------------------
# 3) Evaluation function (same metrics as R)
# ------------------------
def evaluate_model(actual, prob, pred, name):
    cm = confusion_matrix(actual, pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "Model": name,
        "Accuracy": accuracy_score(actual, pred),
        "Precision": precision_score(actual, pred, zero_division=0),
        "Recall": recall_score(actual, pred, zero_division=0),  # Sensitivity
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "F1_Score": f1_score(actual, pred, zero_division=0),
        "AUC_ROC": roc_auc_score(actual, prob)
    }, cm

# ------------------------
# 4) Logistic Regression (GLM summary like R + sklearn for preds)
# ------------------------
# statsmodels GLM on TRAIN for summary/odds ratios/ICs
train_glm = pd.concat([X_train.copy(), y_train.rename("Churn")], axis=1)
for c in cat_features:
    train_glm[c] = train_glm[c].astype(str)

formula = "Churn ~ " + " + ".join([f"C({c})" for c in cat_features] + num_features)
glm = sm.GLM.from_formula(formula, data=train_glm, family=sm.families.Binomial()).fit()

buf = io.StringIO()
buf.write("Logistic Model Summary (statsmodels GLM on TRAIN):\n")
buf.write(str(glm.summary()) + "\n\n")
buf.write("Odds Ratios:\n" + str(np.exp(glm.params)) + "\n\n")
bic = getattr(glm, "bic", np.nan)
buf.write("LogLik, AIC, BIC:\n" + str((glm.llf, glm.aic, bic)) + "\n")
save_txt(buf.getvalue(), "02_logistic_glm_summary.txt")

# sklearn pipeline for consistent preprocessing + predictions
log_model = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
log_model.fit(X_train, y_train)
log_pred_test  = log_model.predict_proba(X_test)[:, 1]
log_class_test = (log_pred_test > 0.5).astype(int)
log_pred_train  = log_model.predict_proba(X_train)[:, 1]
log_class_train = (log_pred_train > 0.5).astype(int)

# ------------------------
# 5) Decision Trees (Gini & Info) with rpart-like regularization + pruning
# ------------------------
# Preprocess once to dense arrays for pruning/plotting
prep_tree = ColumnTransformer(
    transformers=[
        ("cat", make_ohe(), cat_features),
        ("num", "passthrough", num_features),
    ],
    remainder="drop"
)
Xt_train = prep_tree.fit_transform(X_train)
Xt_test  = prep_tree.transform(X_test)
ohe      = prep_tree.named_transformers_["cat"]
feat_names = list(ohe.get_feature_names_out(cat_features)) + list(num_features)

def fit_pruned_tree(criterion):
    base = dict(
        criterion=criterion,
        random_state=123,
        min_samples_split=20,   # rpart-like minsplit
        min_samples_leaf=7      # rpart-like minbucket
    )
    tmp = DecisionTreeClassifier(**base)
    alphas = tmp.cost_complexity_pruning_path(Xt_train, y_train).ccp_alphas
    best_a, best_auc = 0.0, -1.0
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
    for a in alphas:
        m = DecisionTreeClassifier(**base, ccp_alpha=a)
        scores = []
        for tr, va in cv.split(Xt_train, y_train):
            m.fit(Xt_train[tr], y_train.iloc[tr])
            p = m.predict_proba(Xt_train[va])[:, 1]
            scores.append(roc_auc_score(y_train.iloc[va], p))
        mean_auc = float(np.mean(scores))
        if mean_auc > best_auc:
            best_auc, best_a = mean_auc, a
    return DecisionTreeClassifier(**base, ccp_alpha=best_a).fit(Xt_train, y_train)

tree_gini = fit_pruned_tree("gini")
tree_info = fit_pruned_tree("entropy")

tree_pred_test  = tree_gini.predict_proba(Xt_test)[:, 1]
tree_class_test = (tree_pred_test > 0.5).astype(int)
tree_pred_train  = tree_gini.predict_proba(Xt_train)[:, 1]
tree_class_train = (tree_pred_train > 0.5).astype(int)

# ------------------------
# 6) Random Forest (impurity importance; regularized to avoid overfit)
# ------------------------
rf_model = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
        max_depth=10,            # regularize depth
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=123,
        n_jobs=-1
    ))
])
rf_model.fit(X_train, y_train)
rf_pred_test  = rf_model.predict_proba(X_test)[:, 1]
rf_class_test = (rf_pred_test > 0.5).astype(int)
rf_pred_train  = rf_model.predict_proba(X_train)[:, 1]
rf_class_train = (rf_pred_train > 0.5).astype(int)

# ------------------------
# 7) Gradient Boosting (R gbm params)
# ------------------------
gbm_model = Pipeline([
    ("prep", preprocessor),
    ("clf", GradientBoostingClassifier(
        n_estimators=300,        # n.trees
        learning_rate=0.01,      # shrinkage
        max_depth=3,             # interaction.depth
        min_samples_leaf=10,     # n.minobsinnode
        random_state=123
    ))
])
gbm_model.fit(X_train, y_train)
gbm_pred_test  = gbm_model.predict_proba(X_test)[:, 1]
gbm_class_test = (gbm_pred_test > 0.5).astype(int)
gbm_pred_train  = gbm_model.predict_proba(X_train)[:, 1]
gbm_class_train = (gbm_pred_train > 0.5).astype(int)

# ------------------------
# 8) Evaluate all (Test & Train) like R
# ------------------------
metrics_test, metrics_train, cms = [], [], {}
for name, (pt, ct, pr, cr) in {
    "Logistic Regression": (log_pred_test,  log_class_test,  log_pred_train,  log_class_train),
    "Decision Tree":       (tree_pred_test, tree_class_test, tree_pred_train, tree_class_train),
    "Random Forest":       (rf_pred_test,   rf_class_test,   rf_pred_train,   rf_class_train),
    "Gradient Boosting":   (gbm_pred_test,  gbm_class_test,  gbm_pred_train,  gbm_class_train),
}.items():
    mtest, cmtest = evaluate_model(y_test, pt, ct, name)
    mtrain, cmtrain = evaluate_model(y_train, pr, cr, name)
    metrics_test.append(mtest); metrics_train.append(mtrain)
    cms[name] = {"test_confusion_matrix": cmtest.tolist(),
                 "train_confusion_matrix": cmtrain.tolist()}

pd.DataFrame(metrics_test).to_excel(os.path.join(OUTPUT_DIR, "03_metrics_test.xlsx"), index=False)
pd.DataFrame(metrics_train).to_excel(os.path.join(OUTPUT_DIR, "04_metrics_train.xlsx"), index=False)
save_txt(cms, "05_confusion_matrices.json")

print("\n==== Evaluation on Test Set ====\n", pd.DataFrame(metrics_test))
print("\n==== Evaluation on Train Set ====\n", pd.DataFrame(metrics_train))

# ------------------------
# 9) ROC curves (Test)
# ------------------------
plt.figure(figsize=(8,6))
for label, probs in [
    ("Logistic",      log_pred_test),
    ("Decision Tree", tree_pred_test),
    ("Random Forest", rf_pred_test),
    ("GBM",           gbm_pred_test),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
plt.plot([0,1],[0,1], linestyle="--")
plt.title("ROC Curves - Test Set")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
save_fig("06_roc_curves_test.png")

# ------------------------
# 10) Variable importance (RF & GBM)
# ------------------------
def feature_names_from_preproc(prep, numeric_cols, categorical_cols):
    ohe = prep.named_transformers_["cat"]
    return list(ohe.get_feature_names_out(categorical_cols)) + list(numeric_cols)

# RF
rf_clf = rf_model.named_steps["clf"]
rf_feats = feature_names_from_preproc(rf_model.named_steps["prep"], num_features, cat_features)
rf_imp = pd.DataFrame({"Variable": rf_feats, "Importance": rf_clf.feature_importances_}).sort_values("Importance", ascending=False)
rf_imp.head(10).to_excel(os.path.join(OUTPUT_DIR, "07_rf_top10_importance.xlsx"), index=False)
plt.figure(figsize=(10,6))
plt.bar(range(10), rf_imp["Importance"].head(10))
plt.xticks(range(10), rf_imp["Variable"].head(10), rotation=60, ha="right")
plt.title("Top 10 Important Features (Random Forest)")
save_fig("07_rf_top10_importance.png")

# GBM
gbm_clf = gbm_model.named_steps["clf"]
gbm_feats = feature_names_from_preproc(gbm_model.named_steps["prep"], num_features, cat_features)
gbm_imp = pd.DataFrame({"var": gbm_feats, "rel.inf": gbm_clf.feature_importances_}).sort_values("rel.inf", ascending=False)
gbm_imp.head(10).to_excel(os.path.join(OUTPUT_DIR, "08_gbm_top10_importance.xlsx"), index=False)
plt.figure(figsize=(10,6))
plt.bar(range(10), gbm_imp["rel.inf"].head(10))
plt.xticks(range(10), gbm_imp["var"].head(10), rotation=60, ha="right")
plt.title("Top 10 Important Features (GBM)")
save_fig("08_gbm_top10_importance.png")

# ------------------------
# 11) Plot pruned trees (top levels)
# ------------------------
plt.figure(figsize=(18,10))
plot_tree(tree_gini, max_depth=3, feature_names=feat_names, class_names=["No","Yes"], filled=True)
plt.title("Decision Tree (Gini) - Top Levels (pruned)")
save_fig("09_tree_gini.png")

plt.figure(figsize=(18,10))
plot_tree(tree_info, max_depth=3, feature_names=feat_names, class_names=["No","Yes"], filled=True)
plt.title("Decision Tree (Information Gain) - Top Levels (pruned)")
save_fig("10_tree_info.png")

# ------------------------
# 12) Lift curves
# ------------------------
def make_lift(actual, prob, model_name):
    d = pd.DataFrame({"actual": np.asarray(actual), "prob": np.asarray(prob)}).sort_values("prob", ascending=False)
    d["cum_actual"] = d["actual"].cumsum()
    total_pos = d["actual"].sum()
    n = len(d)
    d["perc_data"] = (np.arange(1, n+1))/n
    d["perc_target"] = d["cum_actual"]/(total_pos if total_pos else 1)
    d["lift"] = d["perc_target"]/d["perc_data"]
    d["model"] = model_name
    return d[["perc_data","lift","model"]]

lift_all = pd.concat([
    make_lift(y_test, log_pred_test,  "Logistic Regression"),
    make_lift(y_test, tree_pred_test, "Decision Tree"),
    make_lift(y_test, rf_pred_test,   "Random Forest"),
    make_lift(y_test, gbm_pred_test,  "Gradient Boosting"),
], ignore_index=True)
lift_all.to_excel(os.path.join(OUTPUT_DIR, "11_lift_data.xlsx"), index=False)

plt.figure(figsize=(8,6))
for name, g in lift_all.groupby("model"):
    plt.plot(g["perc_data"], g["lift"], label=name)
plt.axhline(1.0, linestyle="--", color="black")
plt.title("Lift Curves for All Models")
plt.xlabel("Proportion of Test Set (sorted by predicted probability)")
plt.ylabel("Lift")
plt.legend()
save_fig("11_lift_curves.png")

# ------------------------
# 13) KS curves & significance tests
# ------------------------
def ks_plot(prob, y, model_name, fname):
    d = pd.DataFrame({"prob": prob, "label": np.asarray(y)}).sort_values("prob", ascending=False)
    cum_pos = (d["label"].cumsum()/d["label"].sum()) if d["label"].sum()>0 else np.zeros(len(d))
    cum_neg = ((1-d["label"]).cumsum()/(1-d["label"]).sum()) if (1-d["label"]).sum()>0 else np.zeros(len(d))
    ks_series = np.abs(cum_pos - cum_neg)
    ks = ks_series.max()
    ix = int(np.argmax(ks_series.values))

    plt.figure(figsize=(8,6))
    plt.plot(cum_pos.values, label="Positive Class (1)")
    plt.plot(cum_neg.values, label="Negative Class (0)")
    plt.vlines(ix, ymin=cum_neg.values[ix], ymax=cum_pos.values[ix], linestyles="dashed")
    plt.title(f"Kolmogorov–Smirnov Curve - {model_name} (KS={ks:.4f})")
    plt.xlabel("Sorted Observations"); plt.ylabel("Cumulative Distribution")
    plt.legend(loc="lower right")
    save_fig(fname)
    return float(ks)

def ks_test(prob, y):
    p1 = np.asarray(prob)[np.asarray(y)==1]
    p0 = np.asarray(prob)[np.asarray(y)==0]
    return ks_2samp(p1, p0)

ks_out = {}
for name, probs in {
    "Logistic Regression": log_pred_test,
    "Decision Tree":       tree_pred_test,
    "Random Forest":       rf_pred_test,
    "Gradient Boosting":   gbm_pred_test,
}.items():
    ks_curve = ks_plot(probs, y_test, name, f"12_ks_{name.replace(' ','_').lower()}.png")
    ks_res = ks_test(probs, y_test)
    ks_out[name] = {"KS_statistic": float(ks_res.statistic), "p_value": float(ks_res.pvalue), "curve_KS": ks_curve}
save_txt(ks_out, "12_ks_tests.json")

# ------------------------
# 14) Cohen's Kappa
# ------------------------
kappas = {
    "Logistic Regression": cohen_kappa_score(y_test, log_class_test),
    "Decision Tree":       cohen_kappa_score(y_test, tree_class_test),
    "Random Forest":       cohen_kappa_score(y_test, rf_class_test),
    "Gradient Boosting":   cohen_kappa_score(y_test, gbm_class_test),
}
save_txt({k: float(v) for k, v in kappas.items()}, "13_cohens_kappa.json")

print("\n✅ Finished. All artifacts saved to:", OUTPUT_DIR)
