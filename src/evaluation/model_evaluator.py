
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)


class ModelEvaluator:

    @staticmethod
    def evaluate(y_true, y_prob, threshold=0.5):

        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob, dtype=float)

        if len(y_true) == 0:
            raise ValueError("Evaluation data cannot be empty.")

        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have equal lengths.")

        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("Labels must contain only 0 and 1.")

        if not np.isfinite(y_prob).all():
            raise ValueError("Predictions contain NaN or infinity.")

        if np.any((y_prob < 0) | (y_prob > 1)):
            raise ValueError("Probabilities must be between 0 and 1.")

        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")

        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        has_both_classes = len(np.unique(y_true)) == 2

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, y_pred)
            ),
            "precision": float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, zero_division=0)
            ),
            "roc_auc": (
                float(roc_auc_score(y_true, y_prob))
                if has_both_classes else None
            ),
            "pr_auc": (
                float(average_precision_score(y_true, y_prob))
                if has_both_classes else None
            ),
            "threshold": float(threshold),
            "total_samples": int(len(y_true)),
            "fraud_samples": int(np.sum(y_true == 1)),
            "legitimate_samples": int(np.sum(y_true == 0)),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

        metrics["confusion_matrix"] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }

        metrics["classification_report"] = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
            output_dict=True
        )

        return metrics
