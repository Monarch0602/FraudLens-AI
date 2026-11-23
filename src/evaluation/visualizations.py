
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve
)


class EvaluationVisualizer:

    @staticmethod
    def plot_confusion_matrix(metrics):

        cm = metrics["confusion_matrix"]

        matrix = [
            [cm["tn"], cm["fp"]],
            [cm["fn"], cm["tp"]]
        ]

        fig, ax = plt.subplots(figsize=(6, 5))

        image = ax.imshow(matrix, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["Predicted Legitimate", "Predicted Fraud"])
        ax.set_yticklabels(["Actual Legitimate", "Actual Fraud"])

        ax.set_title("Fraud Detection Confusion Matrix")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(matrix[i][j]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold"
                )

        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        return fig

    @staticmethod
    def plot_roc_curve(y_true, y_prob):

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(fpr, tpr, label="FraudLens Model")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        return fig

    @staticmethod
    def plot_precision_recall_curve(y_true, y_prob):

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_prob
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(recall, precision, label="FraudLens Model")

        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()

        return fig

    @staticmethod
    def plot_thresholds(results):

        thresholds = [r["threshold"] for r in results]
        precision = [r["precision"] for r in results]
        recall = [r["recall"] for r in results]
        f1 = [r["f1_score"] for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(thresholds, precision, marker="o", label="Precision")
        ax.plot(thresholds, recall, marker="o", label="Recall")
        ax.plot(thresholds, f1, marker="o", label="F1 Score")

        ax.set_title("Threshold Performance Analysis")
        ax.set_xlabel("Probability Threshold")
        ax.set_ylabel("Score")

        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()

        return fig
