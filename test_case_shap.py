
from pprint import pprint

from src.investigation.case_manager import CaseManager


def main():
    manager = CaseManager()

    case_id = "FL-001315"
    case = manager.get_case(case_id)

    if not case:
        print(f"Case {case_id} not found.")
        return

    print("=" * 60)
    print("FRAUDLENS AI - CASE DATA INSPECTION")
    print("=" * 60)

    print("\nCASE FIELDS:")
    print(list(case.keys()))

    print("\nTRANSACTION DATA:")
    transaction = case.get("transaction_data", {})

    pprint(transaction)

    print("\nTRANSACTION DATA FIELDS:")
    print(list(transaction.keys()))

    print("\nDATA TYPES:")
    for key, value in transaction.items():
        print(f"{key}: {type(value).__name__} -> {value}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
