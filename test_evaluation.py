
import numpy as np
import matplotlib.pyplot as plt

from src.evaluation import (
    ModelEvaluator,
    ThresholdOptimizer,
    EvaluationVisualizer
)


def main():

    # Simulated test data
    y_true = np.array([
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ])

    y_prob = np.array([
        0.05, 0.10, 0.20, 0.30, 0.60,
        0.45, 0.55, 0.70, 0.85, 0.95
    ])

    print("=" * 60)
    print("FRAUDLENS ML EVALUATION TEST")
    print("=" * 60)

    # 1. Test evaluator
    print("\n[1] Testing ModelEvaluator...")

    metrics = ModelEvaluator.evaluate(
        y_true,
        y_prob,
        threshold=0.50
    )

    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1 Score:", metrics["f1_score"])
    print("ROC-AUC:", metrics["roc_auc"])
    print("PR-AUC:", metrics["pr_auc"])
    print("Confusion Matrix:", metrics["confusion_matrix"])

    # 2. Test threshold optimizer
    print("\n[2] Testing ThresholdOptimizer...")

    threshold_results = ThresholdOptimizer.analyze(
        y_true,
        y_prob
    )

    best = ThresholdOptimizer.best_threshold(
        threshold_results,
        metric="f1_score"
    )

    print("Thresholds tested:", len(threshold_results))
    print("Best threshold:", best["threshold"])
    print("Best F1:", best["f1_score"])

    # 3. Test visualizations
    print("\n[3] Testing EvaluationVisualizer...")

    cm_fig = EvaluationVisualizer.plot_confusion_matrix(metrics)
    roc_fig = EvaluationVisualizer.plot_roc_curve(y_true, y_prob)
    pr_fig = EvaluationVisualizer.plot_precision_recall_curve(y_true, y_prob)
    threshold_fig = EvaluationVisualizer.plot_thresholds(threshold_results)

    # Save generated plots
    cm_fig.savefig("test_confusion_matrix.png")
    roc_fig.savefig("test_roc_curve.png")
    pr_fig.savefig("test_precision_recall_curve.png")
    threshold_fig.savefig("test_threshold_analysis.png")

    print("All 4 visualizations generated successfully.")

    plt.close("all")

    print("\n" + "=" * 60)
    print("ALL BASIC EVALUATION TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
