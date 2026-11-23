
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_raw_data
from src.models.splitting import split_three_way
from src.models.training import FraudModelTrainer
from src.explainability.shap_explainer import FraudExplainer


def main():

    print("=" * 70)
    print("FRAUDLENS AI - SHAP EXPLAINABILITY")
    print("=" * 70)

    # Load data
    data = load_raw_data()

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

    # Create output directory
    output_dir = Path("reports/shap")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limit SHAP computation to a sample of test transactions
    sample_size = min(200, len(X_test_processed))

    X_sample = X_test_processed.iloc[:sample_size].copy()

    y_sample = y_test.iloc[:sample_size].copy()

    probabilities = model.predict_proba(X_sample)[:, 1]

    predictions = (probabilities >= 0.85).astype(int)

    print(f"\nAnalyzing {sample_size} test transactions...")

    # Create SHAP explainer
    explainer = FraudExplainer(model)

    # Compute SHAP values
    shap_values = explainer.explain(X_sample)

    # Global importance
    importance_df = explainer.global_importance(
        X_sample,
        shap_values
    )

    importance_df.to_csv(
        output_dir / "global_feature_importance.csv",
        index=False
    )

    print("\nTOP 10 GLOBAL FEATURES")
    print("-" * 50)
    print(importance_df.head(10).to_string(index=False))

    # Generate plots
    explainer.plot_summary(
        X_sample,
        shap_values,
        output_dir
    )

    explainer.plot_bar(
        X_sample,
        shap_values,
        output_dir
    )

    # Choose a flagged transaction for local explanation
    flagged_indices = np.where(predictions == 1)[0]

    if len(flagged_indices) > 0:

        selected_index = int(flagged_indices[0])
        print("\nSelected flagged transaction for explanation.")

    else:

        selected_index = int(np.argmax(probabilities))
        print("\nNo flagged transaction in sample.")
        print("Explaining highest-risk transaction instead.")

    # Local explanation
    explanation_df = explainer.explain_transaction(
        X_sample,
        shap_values,
        selected_index,
        output_dir
    )

    print("\nTRANSACTION EXPLANATION")
    print("-" * 50)

    print(
        f"Test sample index: {selected_index}"
    )

    print(
        f"Fraud probability: "
        f"{probabilities[selected_index]:.4f}"
    )

    print(
        f"Actual label: "
        f"{int(y_sample.iloc[selected_index])}"
    )

    print(
        f"Predicted label: "
        f"{int(predictions[selected_index])}"
    )

    print("\nTOP CONTRIBUTING FEATURES")
    print(
        explanation_df.head(10)[
            [
                "feature",
                "feature_value",
                "shap_value",
                "impact"
            ]
        ].to_string(index=False)
    )

    print("\nSHAP REPORTS GENERATED")
    print("-" * 50)

    for file in sorted(output_dir.iterdir()):
        print(file.name)

    print("\nSHAP analysis completed successfully.")


if __name__ == "__main__":
    main()
