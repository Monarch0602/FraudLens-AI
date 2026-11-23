
from src.data.loader import load_raw_data
from src.data.validation import DataValidator


def main():

    # Load raw transaction data
    data = load_raw_data()

    print("\n" + "=" * 50)
    print("FRAUDLENS AI - DATA VALIDATION")
    print("=" * 50)

    # Run validation
    validator = DataValidator(data)
    report = validator.validate()

    # Print overview
    print("\nDATASET OVERVIEW")
    print(report["dataset_overview"])

    # Print missing values
    print("\nMISSING VALUES")
    print(report["missing_values"])

    print("\nDUPLICATE ROWS")
    print(report["duplicate_rows"])

    # Print data types
    print("\nDATA TYPES")
    print(report["data_types"])

    # Print target distribution
    print("\nTARGET COLUMN")
    print(report["target_column"])

    print("\nTARGET DISTRIBUTION")
    print(report.get("target_distribution", "Not detected"))

    print("\nTARGET PERCENTAGES")
    print(report.get("target_percentage", "Not detected"))

    print("\nCLASS IMBALANCE RATIO")
    print(report.get("class_imbalance_ratio", "Not calculated"))

    # Print warnings
    print("\nWARNINGS")
    for warning in report["warnings"]:
        print(f"- {warning}")

    print("\nVALIDATION STATUS")
    print(report["validation_status"])

    # Save report
    validator.save_report(report)

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
