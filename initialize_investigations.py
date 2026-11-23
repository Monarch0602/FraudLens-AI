
from src.investigation.case_manager import CaseManager


def main():

    print("=" * 60)
    print("FRAUDLENS AI - INVESTIGATION DATABASE")
    print("=" * 60)

    manager = CaseManager()

    print("\nInitializing investigation database...")

    imported = manager.import_transactions()

    print(f"\nNew cases imported: {imported}")

    summary = manager.get_summary()

    print("\nDATABASE SUMMARY")
    print("-" * 40)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nTOP PRIORITY CASES")
    print("-" * 40)

    cases = manager.get_cases(
        flagged_only=True
    )

    if not cases.empty:

        columns = [
            "case_id",
            "transaction_id",
            "amount",
            "category",
            "risk_score",
            "risk_tier",
            "status"
        ]

        print(
            cases[columns]
            .head(10)
            .to_string(index=False)
        )

    print("\nDatabase initialized successfully.")


if __name__ == "__main__":
    main()
