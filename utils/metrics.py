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

    Binary (cm.shape[0]==2): sensitivity/specificity are the usual
    pos_label=0 recall / negative-class recall. Multi-class: sensitivity is
    macro-averaged recall, and specificity is macro-averaged one-vs-rest
    specificity (TN/(TN+FP) per class, then averaged) -- a binary-only
    cm[1,1]/(cm[1,1]+cm[1,0]) formula would silently compute a meaningless
    number for 3+ classes (just class 1's recall restricted to labels 0/1,
    ignoring every other class entirely).
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    cm = confusion_matrix(labels, preds)

    if cm.shape[0] == 2:
        specificity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
    else:
        total = cm.sum()
        specs = []
        for c in range(cm.shape[0]):
            tp = cm[c, c]
            fn = cm[c, :].sum() - tp
            fp = cm[:, c].sum() - tp
            tn = total - tp - fn - fp
            specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        specificity = float(np.mean(specs))

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "sensitivity": recall_score(labels, preds, average="binary", pos_label=0, zero_division=0)
                       if cm.shape[0] == 2 else recall_score(labels, preds, average="macro", zero_division=0),
        "specificity": specificity,
    }


def aggregate_predictions_by_subject(probs, subject_ids, labels):
    """
    Test-time augmentation aggregation: probs (N, n_classes) softmax scores from
    N (possibly multiple-per-subject) windows, subject_ids (N,) the subject each
    row came from, labels (N,) that row's true label (identical across all of one
    subject's rows). Averages probs across each subject's windows, then argmaxes
    the average -- same TTA scheme as the official CAUEEG reference
    implementation (github.com/ipis-mjkim/caueeg-ceednet's check_accuracy_multicrop:
    average softmax over crops, not per-crop majority vote), just averaging over
    however many windows a subject naturally has rather than a fixed crop count.

    Returns (agg_preds, agg_labels), each length = number of UNIQUE subjects, in
    first-occurrence order -- feed these into compute_loso_metrics, not the raw
    per-window probs/preds, or subjects with more windows get counted more than
    once and silently inflate/distort the reported metrics.
    """
    probs = np.asarray(probs)
    by_subject = {}
    order = []
    for i, sid in enumerate(subject_ids):
        if sid not in by_subject:
            by_subject[sid] = {"probs": [], "label": labels[i]}
            order.append(sid)
        by_subject[sid]["probs"].append(probs[i])

    agg_preds, agg_labels = [], []
    for sid in order:
        mean_probs = np.mean(by_subject[sid]["probs"], axis=0)
        agg_preds.append(int(np.argmax(mean_probs)))
        agg_labels.append(by_subject[sid]["label"])
    return agg_preds, agg_labels
