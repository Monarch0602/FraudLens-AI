
from src.data.loader import load_raw_data
from src.features.engineering import FraudFeatureEngineer


def main():

    data = load_raw_data()

    # Separate target
    y = data["fraud"].astype(int)

    # Feature engineering
    engineer = FraudFeatureEngineer()

    features = engineer.fit_transform(data.drop(columns=["fraud"]))

    print("\n" + "=" * 60)
    print("FRAUDLENS AI - FEATURE ENGINEERING")
    print("=" * 60)

    print("\nOriginal feature shape:")
    print(data.drop(columns=["fraud"]).shape)

    print("\nEngineered feature shape:")
    print(features.shape)

    print("\nGenerated features:")
    print(features.columns.tolist())

    print("\nFeature preview:")
    print(features.head().to_string())

    print("\nTarget preview:")
    print(y.head().to_string())

    print("\nFeature missing values:")
    print(features.isnull().sum())

    print("\nFeature engineering test completed.")


if __name__ == "__main__":
    main()
