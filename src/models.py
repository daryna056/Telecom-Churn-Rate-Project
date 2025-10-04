
from __future__ import annotations
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_models(random_state: int = 123) -> Dict[str, object]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=None),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
    }
    return models
