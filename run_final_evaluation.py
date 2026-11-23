
from pathlib import Path
import json

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from src.data.loader import load_raw_data
from src.models.splitting import split_three_way
from src.models.training import FraudModelTrainer


def main():

    print("=" * 70)
    print("FRAUDLENS AI - FINAL TEST EVALUATION")
    print("=" * 70)

    # Load dataset and reproduce the same deterministic split
    data = load_raw_data()

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = split_three_way(data)

    # Fit feature engineering only on training data
    trainer = FraudModelTrainer()

    X_train_processed, X_test_processed = trainer.prepare_features(
        X_train, X_test
    )

    # Select the model based on validation F1
    selected_model = trainer.build_models()["Extra Trees"]

    print("\nSelected model: Extra Trees")
    print("Selected threshold: 0.85")
    print("Selection basis: validation F1")

    # Train on training set only
    print("\nTraining selected model...")
    selected_model.fit(X_train_processed, y_train)

    # Final test probabilities
    probabilities = selected_model.predict_proba(
        X_test_processed
    )[:, 1]

    # Apply the threshold selected on validation data
    threshold = 0.85
    predictions = (probabilities >= threshold).astype(int)

    # Metrics
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "Extra Trees",
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "false_positive_rate": (
            fp / (fp + tn) if (fp + tn) else 0
        )
    }

    print("\nFINAL TEST RESULTS")
    print("=" * 70)

    for key, value in metrics.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nCONFUSION MATRIX")
    print("=" * 40)
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")

    print("\nCLASSIFICATION REPORT")
    print("=" * 70)

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    # Save report
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    with open(
        report_dir / "final_test_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame([metrics]).to_csv(
        report_dir / "final_test_metrics.csv",
        index=False
    )

    print("\nReports saved:")
    print("reports/final_test_metrics.json")
    print("reports/final_test_metrics.csv")

    print("\nFinal test evaluation completed.")


if __name__ == "__main__":
    main()
