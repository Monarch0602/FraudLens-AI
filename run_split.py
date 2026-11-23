
from src.data.loader import load_raw_data
from src.models.splitting import split_data


def main():

    data = load_raw_data()

    X_train, X_test, y_train, y_test = split_data(data)

    print("\nSPLIT VERIFICATION")
    print("=" * 40)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nTrain target shape:", y_train.shape)
    print("Test target shape:", y_test.shape)

    print("\nFraud cases in training:", int(y_train.sum()))
    print("Fraud cases in testing:", int(y_test.sum()))

    print("\nSplit completed successfully.")


if __name__ == "__main__":
    main()
