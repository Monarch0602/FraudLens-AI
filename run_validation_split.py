
from src.data.loader import load_raw_data
from src.models.splitting import split_three_way


def main():

    data = load_raw_data()

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = split_three_way(data)

    print("\nSPLIT SHAPE VERIFICATION")
    print("-" * 45)

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    print("\nFraud counts:")
    print("Train:", int(y_train.sum()))
    print("Validation:", int(y_val.sum()))
    print("Test:", int(y_test.sum()))

    print("\nThree-way split completed successfully.")


if __name__ == "__main__":
    main()
