
from src.data.loader import load_raw_data
from src.data.eda import FraudEDA


def main():

    # Load dataset
    data = load_raw_data()

    # Initialize EDA
    eda = FraudEDA(data)

    # Execute all analysis
    results = eda.run_all()

    print("\n" + "=" * 55)
    print("FRAUDLENS AI - EXPLORATORY DATA ANALYSIS RESULTS")
    print("=" * 55)

    print("\nFRAUD DISTRIBUTION")
    print(results["fraud_distribution"])

    print("\nTRANSACTION AMOUNT ANALYSIS")
    print(results["amount_analysis"])

    print("\nCATEGORY ANALYSIS")
    for category in results["category_analysis"]:
        print(category)

    print("\nDEMOGRAPHIC ANALYSIS")
    print(results["demographic_analysis"])

    print("\nTEMPORAL ANALYSIS")
    print(results["temporal_analysis"])

    print("\nCORRELATION ANALYSIS")
    print(results["correlation_analysis"])

    print("\nEDA pipeline finished.")


if __name__ == "__main__":
    main()
