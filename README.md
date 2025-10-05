# 📈 Telecom Customer Churn Prediction

Predict customer churn using a reproducible, Python-based ML pipeline.  
The project includes data preparation, model training, evaluation, and exportable reports/plots.

---

## ✨ Features
- End-to-end pipeline in Python
- Robust preprocessing:
  - Column normalization, missing-value handling for `TotalCharges`
  - One-hot encoding for categorical features
  - Target construction from enhanced IBM Telco fields (`Churn Label` / `Churn Value`)
  - Leakage protection (drops `ChurnLabel`, `ChurnValue`, `ChurnScore`, `ChurnReason`)
- Models:
  - Logistic Regression
  - Decision Tree (Gini & Entropy)
  - Random Forest
  - Gradient Boosting (GBM-style params)
- Metrics & Diagnostics:
  - Accuracy, Precision (PPV), Recall (Sensitivity), **Specificity**, F1
  - **AUC-ROC**, **Cohen’s Kappa**, **KS statistic**
  - ROC curves, lift curves, feature importances, confusion matrices
- Reproducible outputs saved to `outputs/`

---

## 🗂 Project Structure
```
Telecom-Churn-Rate-Project/
├─ src/
│  └─ run_experiment.py        # Main entry point: trains models & writes outputs
├─ Telco_customer_churn.xlsx   # Dataset (enhanced IBM Telco format)
├─ requirements.txt
├─ outputs/                    # Generated on run
└─ README.md
```

---

## 📦 Requirements
- Python 3.9+ recommended
- See `requirements.txt`:
  - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`, `statsmodels`, `openpyxl`

Install everything:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Run
From the repo root:
```bash
python src/run_experiment.py
```

The script:
1. Loads `Telco_customer_churn.xlsx`
2. Normalizes columns and constructs `Churn` (0/1)
3. Splits data (80/20, stratified, seed=123)
4. Trains Logistic, Decision Trees (gini & entropy), Random Forest, Gradient Boosting
5. Evaluates and writes artifacts to `outputs/`

---

## 📊 Outputs (in `outputs/`)
- `01_split_distributions.txt` — class balance
- `02_logistic_glm_summary.txt` — GLM summary, odds ratios, info criteria
- `03_metrics_test.xlsx` — test metrics
- `04_metrics_train.xlsx` — train metrics
- `05_confusion_matrices.json` — per-model confusion matrices
- `06_roc_curves_test.png` — combined ROC
- (Optional further plots/CSVs if enabled later)

---

## 🧪 Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, Specificity, F1
- **Ranking/Separation**: AUC-ROC, KS
- **Agreement**: Cohen’s Kappa

These reflect both operational performance (F1, Precision/Recall) and risk separation (AUC, KS).

---

## 📁 Dataset Notes
This repository expects the enhanced IBM Telco dataset.  
If your sheet uses:
- `Churn Label` → the pipeline converts to `Churn` (Yes→1, No→0)
- `Churn Value` → used directly when present (0/1)
- The following columns are dropped to prevent leakage if present:
  `ChurnLabel`, `ChurnValue`, `ChurnScore`, `ChurnReason`.

Ensure the file is placed at:
```
Telecom-Churn-Rate-Project/Telco_customer_churn.xlsx
```

---

## 🔧 Configuration
Key settings are defined inside `src/run_experiment.py`:
- Train/test split: `test_size=0.2`, `random_state=123`
- Random Forest: `n_estimators=500`
- Gradient Boosting: `n_estimators=300`, `learning_rate=0.01`, `max_depth=3`, `min_samples_leaf=10`

For faster runs on modest hardware, reduce `n_estimators`.

---

## 🚀 Next Steps
- Hyperparameter tuning (GridSearchCV/Optuna)
- Add XGBoost/LightGBM baselines
- Cross-validation with stratified folds
- Model explainability (SHAP)
- API for scoring (FastAPI) and batch scoring jobs

---

## 📝 License
Include your chosen license in `LICENSE` (e.g., MIT).

---
