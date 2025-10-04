# Telco Churn Prediction (Python)

A professional, production-ready project for predicting churn in telecom services — translated from an R workflow into a modern Python stack.

## Key features
- Clean preprocessing pipeline with `ColumnTransformer`
- Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Robust evaluation: Accuracy, Precision, Recall, Specificity, F1, AUC-ROC, Cohen's Kappa, KS-statistic
- Plots: ROC curves, Lift curves, KS curves, Confusion matrices, Feature importance
- Reproducible train/test split and saved artifacts/metrics

## Project structure
```
telco-churn-python/
├── src/
│   ├── data_processing.py      # loading + preprocessing pipeline
│   ├── models.py               # model factory
│   ├── evaluate.py             # metrics, plots, utilities
│   └── run_experiment.py       # main entry-point
├── outputs/                    # metrics and plots generated after running
├── Telco_customer_churn.xlsx   # dataset (IBM Telco)
├── requirements.txt
├── README.md
└── LICENSE
```

## Quickstart
1. **Create environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run**
   ```bash
   python -m src.run_experiment --seed 123 --test-size 0.2
   ```

3. **Outputs**
   - `outputs/metrics_test.csv` and `outputs/metrics_train.csv`
   - `outputs/roc_curves.png`, `outputs/lift_curves.png`, `outputs/ks_curves.png`
   - `outputs/confusion_matrix_<model>.png`
   - `outputs/feature_importance_<model>.png` (where applicable)

## Business framing
- **Objective**: Improve retention by identifying high-risk customers
- **Cutoff**: Default 0.5 (configurable) — examine Precision/Recall tradeoff for campaign sizing
- **Next**: Integrate predictions into CRM (score all active customers weekly), track intervention outcomes, and retrain quarterly

## Notes
- Dataset is included as `Telco_customer_churn.xlsx` (uploaded by you).
- This project prefers standard libraries (no GPU/Cloud).
- Extendable to XGBoost/LightGBM if needed.
