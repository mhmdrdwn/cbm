import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


def compute_loso_metrics(preds, labels):
    """
    Scalar-only metrics for binary/multi-class evaluation: accuracy,
    weighted F1, sensitivity, specificity. Every value here is a plain
    float, safe to format directly.

    preds, labels: 1D array-like of int class labels (label convention:
                   0 = pathology/positive class, matching this project's
                   "pathology first" convention elsewhere).
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "sensitivity": recall_score(labels, preds, average="binary", pos_label=0, zero_division=0)
                       if cm.shape[0] == 2 else recall_score(labels, preds, average="macro", zero_division=0),
        "specificity": (cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm.shape[0] > 1 and (cm[1, 1] + cm[1, 0]) > 0 else 0.0),
    }
