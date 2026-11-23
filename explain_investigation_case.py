
from pathlib import Path

import pandas as pd

from src.data.loader import load_raw_data
from src.models.splitting import split_three_way
from src.models.training import FraudModelTrainer
from src.explainability.shap_explainer import FraudExplainer
from src.investigation.case_manager import CaseManager


def main():

    case_id = "FL-001315"

    output_dir = Path("reports/shap/cases") / case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"FRAUDLENS AI - CASE SHAP EXPLANATION | {case_id}")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Retrieve investigation case
    # --------------------------------------------------

    manager = CaseManager()
    case = manager.get_case(case_id)

    if not case:
        raise ValueError(f"Case {case_id} not found.")

    transaction = case["transaction_data"]

    print("\nCase retrieved successfully.")

    # --------------------------------------------------
    # 2. Load original dataset and recreate data split
    # --------------------------------------------------

    data = load_raw_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_three_way(data)

    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Test rows: {len(X_test)}")

    # --------------------------------------------------
    # 3. Reconstruct the original transaction
    # --------------------------------------------------

    case_raw = pd.DataFrame([{
        "step": transaction["step"],
        "age": transaction["age"],
        "gender": transaction["gender"],
        "category": transaction["category"],
        "amount": transaction["amount"]
    }])

    print("\nCase transaction reconstructed.")
    print(case_raw.to_string(index=False))

    # --------------------------------------------------
    # 4. Fit feature engineering on training data only
    # --------------------------------------------------

    trainer = FraudModelTrainer()

    X_train_processed, X_case_processed = trainer.prepare_features(
        X_train,
        case_raw
    )

    print("\nFeature engineering completed.")
    print(f"Processed feature count: {X_case_processed.shape[1]}")

    # --------------------------------------------------
    # 5. Train the selected model
    # --------------------------------------------------

    model = trainer.build_models()["Extra Trees"]

    print("\nTraining Extra Trees model...")

    model.fit(X_train_processed, y_train)

    # --------------------------------------------------
    # 6. Verify model prediction
    # --------------------------------------------------

    probability = model.predict_proba(X_case_processed)[0, 1]
    prediction = int(probability >= 0.85)

    print("\nCASE MODEL RESULTS")
    print("-" * 50)
    print(f"Case ID: {case_id}")
    print(f"Fraud probability: {probability:.6f}")
    print(f"Predicted label: {prediction}")
    print(f"Existing case probability: {case['fraud_probability']:.6f}")

    # --------------------------------------------------
    # 7. Generate SHAP explanation
    # --------------------------------------------------

    print("\nCalculating SHAP values for this case...")

    explainer = FraudExplainer(model)

    shap_values = explainer.explain(X_case_processed)

    explanation_df = explainer.explain_transaction(
        X_case_processed,
        shap_values,
        0,
        output_dir
    )

    # --------------------------------------------------
    # 8. Save explanation
    # --------------------------------------------------

    explanation_path = output_dir / "case_shap_explanation.csv"

    explanation_df.to_csv(
        explanation_path,
        index=False
    )

    print("\nTOP CONTRIBUTING FEATURES")
    print("-" * 50)

    display_columns = [
        "feature",
        "feature_value",
        "shap_value",
        "impact"
    ]

    print(
        explanation_df.head(10)[display_columns]
        .to_string(index=False)
    )

    print("\nOUTPUT FILES")
    print("-" * 50)

    for file in sorted(output_dir.iterdir()):
        print(file.name)

    print("\nCase-specific SHAP analysis completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
