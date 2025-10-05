# ======================
# Telco Churn Prediction — Python mirror of the R project
# ======================

import os
import io
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
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
