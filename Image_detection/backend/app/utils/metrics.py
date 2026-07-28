from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None
) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics for model evaluation."""
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    auc = None
    if y_probs is not None:
        try:
            if y_probs.ndim > 1 and y_probs.shape[1] == 2:
                auc = float(roc_auc_score(y_true, y_probs[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["Real", "AI Generated"], output_dict=True, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }
