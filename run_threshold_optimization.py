
from src.data.loader import load_raw_data
from src.models.splitting import split_three_way
from src.models.training import FraudModelTrainer
from src.models.threshold_optimizer import ThresholdOptimizer


def main():

    print("=" * 80)
    print("FRAUDLENS AI - THRESHOLD OPTIMIZATION")
    print("=" * 80)

    # Load data
    data = load_raw_data()

    # Create train / validation / test split
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = split_three_way(data)

    # Feature engineering is fitted only on training data
    trainer = FraudModelTrainer()

    X_train_processed, X_val_processed = trainer.prepare_features(
        X_train, X_val
    )

    # Build models
    models = trainer.build_models()

    optimizer = ThresholdOptimizer()

    print("\nTRAINING MODELS ON TRAINING DATA")
    print("=" * 80)

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train_processed, y_train)

        # Probability predictions on validation set only
        probabilities = model.predict_proba(X_val_processed)[:, 1]

        result = optimizer.optimize_model(
            model_name,
            y_val,
            probabilities
        )

        best = result["best_f1"]

        print("\nBest F1 threshold:")
        print(f"Threshold: {best['threshold']:.2f}")
        print(f"Precision: {best['precision']:.4f}")
        print(f"Recall: {best['recall']:.4f}")
        print(f"F1: {best['f1']:.4f}")
        print(f"False positives: {best['false_positive']}")
        print(f"False negatives: {best['false_negative']}")

    # Save all threshold experiments
    optimizer.save_results()

    # Display comparison
    optimizer.print_summary()

    print("\nThreshold optimization completed.")
    print("Validation data was used for threshold selection.")
    print("Test data was NOT used.")


if __name__ == "__main__":
    main()
