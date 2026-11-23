
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class ThresholdOptimizer:

    @staticmethod
    def analyze(y_true, y_prob, thresholds=None):

        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob, dtype=float)

        if len(y_true) == 0 or len(y_true) != len(y_prob):
            raise ValueError("Invalid evaluation data.")

        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("Labels must contain only 0 and 1.")

        if not np.isfinite(y_prob).all():
            raise ValueError("Predictions contain NaN or infinity.")

        if np.any((y_prob < 0) | (y_prob > 1)):
            raise ValueError("Probabilities must be between 0 and 1.")

        if thresholds is None:
            thresholds = np.arange(0.05, 1.00, 0.05)

        results = []

        for threshold in thresholds:

            threshold = float(threshold)

            y_pred = (y_prob >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            ).ravel()

            results.append({
                "threshold": round(threshold, 3),
                "precision": float(
                    precision_score(y_true, y_pred, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_true, y_pred, zero_division=0)
                ),
                "f1_score": float(
                    f1_score(y_true, y_pred, zero_division=0)
                ),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "flagged_transactions": int(tp + fp)
            })

        return results

    @staticmethod
    def best_threshold(results, metric="f1_score"):

        if not results:
            raise ValueError("Threshold results cannot be empty.")

        if metric not in {"precision", "recall", "f1_score"}:
            raise ValueError("Unsupported optimization metric.")

        return max(results, key=lambda item: item[metric])
