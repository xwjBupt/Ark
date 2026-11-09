import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, precision_recall_curve, auc
from data import CLASSES


def evaluate(labels, probabilities, thresholds=None):
    per_class = []
    for i, name in enumerate(CLASSES):
        y, p = labels[:, i], probabilities[:, i]
        valid = len(np.unique(y)) == 2
        precision, recall, _ = precision_recall_curve(y, p)
        per_class.append({"id": i+1, "name": name, "positives": int(y.sum()),
                          "auc": float(roc_auc_score(y, p)) if valid else None,
                          "ap": float(average_precision_score(y, p)) if y.sum() else None,
                          "pr_auc": float(auc(recall, precision)) if y.sum() else None})
    output = {"macro_auc": float(np.mean([c["auc"] for c in per_class if c["auc"] is not None])),
              "macro_ap": float(np.mean([c["ap"] for c in per_class if c["ap"] is not None])),
              "macro_pr_auc": float(np.mean([c["pr_auc"] for c in per_class if c["pr_auc"] is not None])),
              "macro_f1_05": float(f1_score(labels, probabilities >= 0.5, average="macro", zero_division=0)),
              "per_class": per_class}
    if thresholds is not None:
        output["macro_f1_tuned"] = float(f1_score(labels, probabilities >= np.asarray(thresholds), average="macro", zero_division=0))
    return output


def fit_thresholds(labels, probabilities):
    grid = np.linspace(0.05, 0.95, 37)
    thresholds = []
    for i in range(10):
        scores = [f1_score(labels[:, i], probabilities[:, i] >= t, zero_division=0) for t in grid]
        candidates = np.flatnonzero(np.isclose(scores, max(scores)))
        best = min(candidates, key=lambda j: abs(grid[j]-0.5))
        thresholds.append(float(grid[best]))
    return thresholds
