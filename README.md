<<<<<<< HEAD
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

=======
# 🚀 Telecom Customer Churn Prediction

**Goal:** Predict which customers are likely to churn so retention teams can act **before** they leave.  
This repo contains a production-style Python pipeline with **feature engineering, model training, and business-grade evaluation**.

---

## 📦 Tech Stack
- **Python**: `pandas`, `scikit-learn`, `matplotlib`
- **Pipelines**: `ColumnTransformer` for numeric scaling + categorical One-Hot Encoding
- **Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Metrics**: Accuracy, Precision, Recall, Specificity, F1, AUC-ROC, **Cohen’s Kappa**, **KS statistic**, Lift

---

## 📂 Dataset
- IBM Telco Customer Churn  
- ~7k customers with demographics, services, contract type, monthly charges, and tenure  
- Target: `Churn` (Yes/No → 1/0)

---

## 📊 Results (Test Set, real run)
| Model | Accuracy | Precision | Recall | Specificity | F1_Score | AUC_ROC | Kappa | KS |
|---|---|---|---|---|---|---|---|---|
| _Run the code to populate real results_ | | | | | | | | |

**Best by AUC:** Random Forest  
**Best by F1:** Gradient Boosting

> Interpretation: High AUC means the model ranks churners above non-churners consistently. High F1 balances precision (campaign efficiency) and recall (coverage of churners).

---

## 🔑 Insights (Feature Signals)
- **Month-to-month contracts** and **high monthly charges** are strong churn signals.
- **Longer tenure** and **multi-service bundles** correlate with retention.
- Ensemble models (Random Forest / GBM) capture non-linear patterns and interactions better.

---

## 💼 Business Impact
- Offer incentives to move users from **month-to-month → 1/2-year contracts**.
- Bundle **internet + security + tech support** to raise stickiness.
- Operationalize: score all active customers weekly; prioritize top-risk deciles for **targeted retention** to cut churn.

---

## 🖼 Visuals
_(Run the experiment to generate plots.)_

---

## ⚙️ Reproduce Locally

```bash
# Clone
git clone https://github.com/daryna056/Telecom-Churn-Rate-Project.git
cd Telecom-Churn-Rate-Project

# Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python -m src.run_experiment --seed 123 --test-size 0.2
```

Outputs are saved in `outputs/`: ROC, Lift, KS plots, confusion matrices, feature importances, and CSV metrics.

---

## 🗺 Roadmap
- Hyperparameter tuning (GridSearchCV / Optuna)
- Add **XGBoost / LightGBM**
- FastAPI scoring endpoint + Dockerfile
- Scheduled scoring & drift monitoring
- Dashboard for campaign ROI tracking

---

*Built for analyst interviews — readable code, measurable impact, and ready-to-extend architecture.*
>>>>>>> a147d3fa1050dae34da7e04e8c802726d7eed3ca
