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
