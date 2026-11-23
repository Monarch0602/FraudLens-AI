
from pathlib import Path
import json

import pandas as pd

from src.data.loader import load_raw_data
from src.models.splitting import split_three_way
from src.models.training import FraudModelTrainer
from src.risk.risk_scoring import RiskScoringEngine


def main():

    print("=" * 70)
    print("FRAUDLENS AI - RISK SCORING ENGINE")
    print("=" * 70)

    # Load dataset
    data = load_raw_data()

    # Reproduce our data split
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = split_three_way(data)

    # Prepare features
    trainer = FraudModelTrainer()

    X_train_processed, X_test_processed = trainer.prepare_features(
        X_train, X_test
    )

    # Train selected model
    model = trainer.build_models()["Extra Trees"]

    print("\nTraining Extra Trees...")
    model.fit(X_train_processed, y_train)

    # Predict fraud probabilities
    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    # Initialize scoring engine
    engine = RiskScoringEngine(
        review_threshold=0.85
    )

    # Score transactions
    scored_data = engine.score_transactions(
        probabilities,
        transaction_data=X_test.reset_index(drop=True)
    )

    # Add actual labels for research evaluation only
    scored_data["actual_fraud"] = (
        y_test.reset_index(drop=True).astype(int)
    )

    # Summary
    summary = engine.summary(scored_data)

    # Output directory
    output_dir = Path("reports/risk")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all scores
    scored_data.to_csv(
        output_dir / "transaction_risk_scores.csv",
        index=False
    )

    # Save summary
    with open(
        output_dir / "risk_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(summary, f, indent=4)

    print("\nRISK SUMMARY")
    print("=" * 50)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nRISK TIER DISTRIBUTION")
    print("=" * 50)

    print(
        scored_data["risk_tier"]
        .value_counts()
        .to_string()
    )

    print("\nTOP 15 HIGHEST RISK TRANSACTIONS")
    print("=" * 70)

    display_columns = [
        "amount",
        "category",
        "fraud_probability",
        "risk_score",
        "risk_tier",
        "review_flag",
        "recommended_action",
        "actual_fraud"
    ]

    print(
        scored_data.sort_values(
            "risk_score",
            ascending=False
        )[display_columns]
        .head(15)
        .to_string(index=False)
    )

    print("\nREPORTS SAVED")
    print("-" * 40)
    print("reports/risk/transaction_risk_scores.csv")
    print("reports/risk/risk_summary.json")

    print("\nRisk scoring completed successfully.")


if __name__ == "__main__":
    main()
