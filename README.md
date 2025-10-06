# 📈 Telecom Customer Churn Prediction

A complete, reproducible **Python-based machine learning pipeline** for predicting customer churn using the classic **IBM Telco dataset**.  
The project mirrors an end-to-end analytical workflow — from data cleaning and model training to evaluation, interpretability, and exportable diagnostics.

---

## ✨ Highlights

- End-to-end **modeling workflow**: from preprocessing → model training → evaluation → reporting.
- **Automatic generation of outputs** (plots, metrics, summaries) under `/outputs`.
- **Reproducible design**: fixed seed, stratified splits, no data leakage.
- **Comprehensive evaluation suite** including GLM summaries, ROC/Lift/KS visualizations, and statistical tests.

---

## 🗂 Project Structure
```
Telecom-Churn-Rate-Project/
├─ src/
│  └─ run_experiment.py         # Main experiment script
├─ outputs/                     # Automatically generated artifacts
│   ├─ 01_split_distributions.txt      # Churn class balance in train/test
│   ├─ 02_logistic_glm_summary.txt     # GLM summary, odds ratios, AIC/BIC
│   ├─ 03_metrics_test.xlsx            # Evaluation metrics on test set
│   ├─ 04_metrics_train.xlsx           # Evaluation metrics on train set
│   ├─ 05_confusion_matrices.json      # Confusion matrices (train/test)
│   ├─ 06_roc_curves_test.png          # Combined ROC curves for all models
│   ├─ 07_rf_top10_importance.png      # Random Forest feature importance
│   ├─ 08_gbm_top10_importance.png     # GBM feature importance
│   ├─ 09_tree_gini.png                # Gini-based Decision Tree visualization
│   ├─ 10_tree_info.png                # Entropy-based Decision Tree visualization
│   ├─ 11_lift_curves.png              # Lift curves across models
│   ├─ 12_ks_*.png                     # KS curves per model
│   ├─ 12_ks_tests.json                # KS test results (statistic, p-value)
│   ├─ 13_cohens_kappa.json            # Cohen’s Kappa per model
│   └─ ...                             # Supporting .xlsx and .txt reports
├─ WA_Fn-UseC_-Telco-Customer-Churn.csv
├─ requirements.txt
└─ README.md
```

---

## 📦 Requirements

- **Python 3.9+**
- Libraries:  
  `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`, `statsmodels`, `openpyxl`

Install them via:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ How to Run

From the repo root:

```bash
python src/run_experiment.py
```

The script:

1. Loads `WA_Fn-UseC_-Telco-Customer-Churn.csv`
2. Drops `customerID`, coerces `TotalCharges` to numeric
3. Converts categorical features to one-hot encoded variables
4. Splits data 80/20 (stratified, seed = 123)
5. Trains the following models:
   - **Logistic Regression**
   - **Decision Tree (Gini & Entropy)** with pruning
   - **Random Forest** (500 trees, controlled depth)
   - **Gradient Boosting** (GBM-style)
6. Evaluates all models on both train and test sets
7. Saves metrics, plots, and statistical diagnostics in `/outputs`

---

## 📊 Key Outputs Explained

| File | Description |
|------|--------------|
| **01_split_distributions.txt** | Train/test churn distribution (ensures balanced split) |
| **02_logistic_glm_summary.txt** | Full GLM summary (coefficients, p-values, odds ratios, AIC/BIC) |
| **03_metrics_test.xlsx** | Test-set metrics (Accuracy, Precision, Recall, Specificity, F1, AUC) |
| **04_metrics_train.xlsx** | Same metrics on training data for overfitting check |
| **06_roc_curves_test.png** | Combined ROC curves with per-model AUC |
| **11_lift_curves.png** | Lift curves for all models — visualize campaign efficiency |
| **12_ks_tests.json** | KS statistic and p-values for distributional separation |
| **13_cohens_kappa.json** | Kappa coefficients quantifying classifier agreement |

All other `.png` and `.xlsx` files correspond to feature importances and decision tree visualizations.

---

## 🧪 Evaluation Metrics

- **Classification Quality:** Accuracy, Precision, Recall, Specificity, F1-score  
- **Ranking Performance:** AUC-ROC, KS statistic  
- **Agreement:** Cohen’s Kappa  

These metrics together show **predictive performance**, **class separation strength**, and **model reliability**.

---

## 🔍 Insights from the Outputs

- **ROC Curves:**  
  Random Forest and Gradient Boosting dominate by AUC (~0.85–0.88), confirming superior ranking ability.
  
- **Feature Importances:**  
  `Contract`, `tenure`, and `MonthlyCharges` emerge as top drivers — consistent with domain intuition.
  
- **Lift Curves:**  
  Ensemble models show >2× lift in the top 20% of customers, useful for targeting retention campaigns.
  
- **KS Tests:**  
  All models achieve statistically significant KS (>0.4), indicating clear separation between churners and non-churners.
  
- **Kappa:**  
  Cohen’s Kappa confirms strong agreement (~0.55–0.65), reducing concern of random predictions.

---

## ⚙️ Model Settings (Key Parameters)

| Model | Key Parameters |
|--------|----------------|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `min_samples_split=20`, `min_samples_leaf=7`, automatic pruning |
| Random Forest | `n_estimators=500`, `max_depth=10`, `class_weight=balanced` |
| Gradient Boosting | `n_estimators=300`, `learning_rate=0.01`, `max_depth=3`, `min_samples_leaf=10` |

These settings replicate **R’s default caret / ranger / gbm** configurations while controlling overfitting.

---

## 🚀 Next Steps

- Add **cross-validation** and **hyperparameter tuning** (`GridSearchCV`, `Optuna`)
- Integrate **LightGBM/XGBoost**
- Add **SHAP interpretability** layer
- Deploy scoring API via **FastAPI + Docker**
- Create automated **report generation** (PDF/HTML)

---

## 📝 License

Licensed under the MIT License (see `LICENSE`).

---

**Author:** [daryna056]  
*Data-driven modeling for measurable business impact.*
