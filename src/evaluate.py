
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, cohen_kappa_score
)

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def evaluate_all(y_true, proba, pred, model_name) -> Dict[str, float]:
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Specificity": specificity_score(y_true, pred),
        "F1_Score": f1_score(y_true, pred, zero_division=0),
        "AUC_ROC": roc_auc_score(y_true, proba) if proba is not None else np.nan,
        "Kappa": cohen_kappa_score(y_true, pred)
    }
    # KS statistic (on scores for positive vs negative)
    pos_scores = proba[y_true == 1]
    neg_scores = proba[y_true == 0]
    if len(pos_scores) and len(neg_scores):
        # empirical CDF difference maximum
        ks = ks_statistic(pos_scores, neg_scores)
    else:
        ks = np.nan
    metrics["KS"] = ks
    return metrics

def ks_statistic(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    # compute KS via sorted CDFs
    all_scores = np.sort(np.concatenate([pos_scores, neg_scores]))
    cdf_pos = np.searchsorted(np.sort(pos_scores), all_scores, side='right') / len(pos_scores)
    cdf_neg = np.searchsorted(np.sort(neg_scores), all_scores, side='right') / len(neg_scores)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))

def plot_roc(all_fprs_tprs: Dict[str, Tuple[np.ndarray, np.ndarray]], savepath: str):
    plt.figure()
    for name, (fpr, tpr) in all_fprs_tprs.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()

def plot_confusion(cm: np.ndarray, classes: List[str], title: str, savepath: str):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()

def lift_curve(y_true: np.ndarray, proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-proba)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = np.sum(y_true)
    perc_data = np.arange(1, len(y_true)+1) / len(y_true)
    perc_target = cum_pos / total_pos if total_pos > 0 else np.zeros_like(cum_pos)
    lift = np.divide(perc_target, perc_data, out=np.zeros_like(perc_data), where=perc_data>0)
    return perc_data, lift

def plot_lift_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], savepath: str):
    plt.figure()
    for name, (x, y) in curves.items():
        plt.plot(x, y, label=name)
    plt.axhline(1.0, linestyle='--')
    plt.xlabel("Proportion of Test Set (sorted by score)")
    plt.ylabel("Lift")
    plt.title("Lift Curves")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()

def plot_ks_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]], savepath: str):
    plt.figure()
    for name, (cum_pos, cum_neg, ks_val) in curves.items():
        plt.plot(cum_pos, label=f"{name} - Pos CDF")
        plt.plot(cum_neg, label=f"{name} - Neg CDF")
    plt.xlabel("Sorted Observations")
    plt.ylabel("Cumulative Distribution")
    plt.title("KS Curves (Test Set)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()

def ks_curve_arrays(y_true: np.ndarray, proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-proba)
    y_sorted = y_true[order]
    pos = (y_sorted == 1).astype(int)
    neg = (y_sorted == 0).astype(int)
    cum_pos = np.cumsum(pos) / max(1, pos.sum())
    cum_neg = np.cumsum(neg) / max(1, neg.sum())
    ks_val = float(np.max(np.abs(cum_pos - cum_neg)))
    return cum_pos, cum_neg, ks_val

def plot_feature_importance(importances, feature_names, title, savepath, topn=10):
    if importances is None or feature_names is None:
        return
    idx = np.argsort(importances)[::-1][:topn]
    plt.figure()
    plt.bar(range(len(idx)), importances[idx])
    plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
