
from src.data.loader import load_raw_data
from src.models.splitting import split_data
from src.models.training import FraudModelTrainer


def main():

    print("=" * 70)
    print("FRAUDLENS AI - MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # Load data
    data = load_raw_data()

    # Split before learning feature statistics
    X_train, X_test, y_train, y_test = split_data(data)

    # Initialize trainer
    trainer = FraudModelTrainer()

    # Train and evaluate models
    results = trainer.train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nTraining pipeline completed.")
    print("Comparison file: reports/model_comparison.csv")


if __name__ == "__main__":
    main()
