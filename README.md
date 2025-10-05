# Telco Customer Churn — Python (R-aligned)

This repo reproduces your R pipeline in Python with the same preprocessing, models, metrics, and plots.

## What’s inside
- **src/** code
  - `data_processing.py` — drop `customerID`, convert `TotalCharges`, encode `Churn` (Yes→1/0), factorize strings.
  - `models.py` — Logistic, Decision Tree *(gini & entropy)*, Random Forest, Gradient Boosting *(R-like params)*.
  - `evaluation.py` — Accuracy, Precision, Recall, **Specificity**, F1, **AUC**, **Kappa**, **KS**, Lift helpers.
  - `run_experiment.py` — stratified 80/20 split (seed=123), trains all models, saves metrics and plots.
- **requirements.txt** — dependable versions.
- **outputs/** — created on run with metrics CSV + ROC/Lift/KS/Importance PNGs.

## Setup (Mac, from scratch)

```bash
# 1) Make a new folder anywhere (or use Downloads)
mkdir -p ~/Telecom-Churn-Rate-Project && cd ~/Telecom-Churn-Rate-Project

# 2) Unzip the project you downloaded
unzip telco-churn-python.zip -d .

# 3) Put the dataset file next to the project root (choose one):
#    a) IBM CSV:
#       WA_Fn-UseC_-Telco-Customer-Churn.csv
#    b) Your Excel:
#       Telco_customer_churn.xlsx

# 4) Create venv + install deps
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5) Run (CSV default) — change --data to your actual filename if different
python src/run_experiment.py --data WA_Fn-UseC_-Telco-Customer-Churn.csv
# or
python src/run_experiment.py --data Telco_customer_churn.xlsx
```

**Outputs will appear in `outputs/`:**
- `metrics_train.csv`, `metrics_test.csv`, `metrics_ks.csv`
- `roc_curves.png`, `lift_curves.png`
- `ks_<Model>.png` (one per model)
- `feature_importance_Random_Forest.png`, `feature_importance_Gradient_Boosting.png`

## GitHub (push to a new repo)

```bash
# inside the project root
git init
git add .
git commit -m "Initial commit: R-aligned Python churn pipeline"
# create a repo on GitHub (e.g., Telecom-Churn-Rate-Project) and then:
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/Telecom-Churn-Rate-Project.git
git push -u origin main
```

## Notes on parity with R
- Split is stratified 80/20 with `random_state=123` (like `createDataPartition`).
- Categorical features are one-hot encoded without dropping the first level, mimicking R factor expansion.
- Models use parameters comparable to your R code:
  - Logistic Regression (`lbfgs`, max_iter=1000)
  - Decision Tree (`gini`, and a second `entropy` tree)
  - Random Forest (500 trees, `gini`)
  - Gradient Boosting (300 trees, learning_rate=0.01, depth=3, min_samples_leaf=10)
- Metrics: Accuracy, Precision (PPV), Recall (Sensitivity), **Specificity**, F1, **AUC**, **Cohen’s Kappa**, **KS**.
- Plots: Combined ROC & Lift; KS curves per model; Feature importances for RF & GBM.

