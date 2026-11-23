
from sklearn.model_selection import train_test_split


def split_data(data, target="fraud", test_size=0.20, random_state=42):

    X = data.drop(columns=[target])
    y = data[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("\nDATA SPLITTING REPORT")
    print("-" * 40)
    print(f"Total samples: {len(data):,}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")
    print(f"Training fraud distribution: {y_train.value_counts().to_dict()}")
    print(f"Testing fraud distribution: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def split_three_way(
    data,
    target="fraud",
    test_size=0.20,
    validation_size=0.20,
    random_state=42
):

    X = data.drop(columns=[target])
    y = data[target].astype(int)

    # First reserve the final test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Validation fraction relative to the remaining data
    relative_validation_size = validation_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=y_train_val
    )

    print("\nTHREE-WAY DATA SPLIT")
    print("=" * 45)

    for name, target_split in [
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test)
    ]:
        print(f"\n{name}: {len(target_split):,} samples")
        print(f"Fraud distribution: {target_split.value_counts().to_dict()}")
        print(f"Fraud rate: {target_split.mean() * 100:.2f}%")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )
