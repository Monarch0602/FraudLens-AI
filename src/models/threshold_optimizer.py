
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


class ThresholdOptimizer:

    def __init__(self, thresholds=None):

        if thresholds is None:
            thresholds = np.arange(0.01, 1.00, 0.01)

        self.thresholds = thresholds
        self.results = {}

    def evaluate_threshold(self, y_true, probabilities, threshold):

        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_true, predictions, zero_division=0
        )

        recall = recall_score(
            y_true, predictions, zero_division=0
        )

        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        fpr = fp / (fp + tn) if (fp + tn) else 0

        return {
            "threshold": round(float(threshold), 3),
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn)
        }

    def optimize_model(self, model_name, y_true, probabilities):

        rows = []

        for threshold in self.thresholds:

            metrics = self.evaluate_threshold(
                y_true,
                probabilities,
                threshold
            )

            metrics["model"] = model_name
            rows.append(metrics)

        results_df = pd.DataFrame(rows)

        # Best F1 operating point
        best_f1 = results_df.loc[
            results_df["f1"].idxmax()
        ].copy()

        # Lowest threshold that achieves at least 90% recall
        high_recall_candidates = results_df[
            results_df["recall"] >= 0.90
        ]

        if not high_recall_candidates.empty:

            high_recall = high_recall_candidates.sort_values(
                by="threshold",
                ascending=True
            ).iloc[0].copy()

        else:
            high_recall = None

        self.results[model_name] = {
            "thresholds": results_df,
            "best_f1": best_f1,
            "high_recall": high_recall
        }

        return self.results[model_name]

    def save_results(self):

        all_results = []

        for model_name, result in self.results.items():

            threshold_df = result["thresholds"].copy()
            all_results.append(threshold_df)

        if not all_results:
            return

        combined = pd.concat(
            all_results,
            ignore_index=True
        )

        output_path = REPORT_DIR / "threshold_analysis.csv"

        combined.to_csv(output_path, index=False)

        print(f"\nThreshold analysis saved to: {output_path}")

    def print_summary(self):

        print("\n" + "=" * 80)
        print("THRESHOLD OPTIMIZATION SUMMARY")
        print("=" * 80)

        for model_name, result in self.results.items():

            print(f"\nMODEL: {model_name}")

            best = result["best_f1"]

            print("\nBEST F1 THRESHOLD")
            print(f"Threshold: {best['threshold']:.2f}")
            print(f"Precision: {best['precision']:.4f}")
            print(f"Recall:    {best['recall']:.4f}")
            print(f"F1:       {best['f1']:.4f}")
            print(f"FP:       {best['false_positive']}")
            print(f"FN:       {best['false_negative']}")

            high_recall = result["high_recall"]

            if high_recall is not None:

                print("\nHIGH-RECALL OPERATING POINT (Recall >= 90%)")
                print(f"Threshold: {high_recall['threshold']:.2f}")
                print(f"Precision: {high_recall['precision']:.4f}")
                print(f"Recall:    {high_recall['recall']:.4f}")
                print(f"F1:       {high_recall['f1']:.4f}")
                print(f"FP:       {high_recall['false_positive']}")
                print(f"FN:       {high_recall['false_negative']}")
            else:
                print("\nNo threshold achieved 90% recall.")
