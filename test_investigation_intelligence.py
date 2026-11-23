
from pathlib import Path
import json

from src.investigation.case_manager import CaseManager
from src.investigation.intelligence.investigation_engine import (
    InvestigationIntelligenceEngine,
)


def main():
    print("=" * 60)
    print("FRAUDLENS AI - INVESTIGATION INTELLIGENCE TEST")
    print("=" * 60)

    # Load case database
    manager = CaseManager()
    cases = manager.get_cases(flagged_only=True)

    if cases.empty:
        print("No flagged cases found.")
        return

    # Select one flagged case
    selected_case_id = cases.iloc[0]["case_id"]

    print(f"\nSelected Case: {selected_case_id}")

    # Retrieve complete case details
    case = manager.get_case(selected_case_id)

    if not case:
        print("Could not retrieve case details.")
        return

    # Initialize intelligence engine
    engine = InvestigationIntelligenceEngine()

    # Generate report (SHAP evidence optional for this test)
    report = engine.generate_report(case)

    # Save report
    output_path = Path("reports/investigations") / (
        f"{selected_case_id}_investigation.json"
    )

    engine.save_report(report, output_path)

    # Display results
    print("\n" + "-" * 60)
    print("RISK ASSESSMENT")
    print("-" * 60)

    print(json.dumps(report["risk_assessment"], indent=4))

    print("\n" + "-" * 60)
    print("OBSERVED EVIDENCE")
    print("-" * 60)

    for item in report["observed_evidence"]:
        print(
            f'{item["description"]}: {item["value"]}'
        )

    print("\n" + "-" * 60)
    print("INVESTIGATION REASONING")
    print("-" * 60)

    for reason in report["reasoning"]:
        print(f"- {reason}")

    print("\n" + "-" * 60)
    print("RECOMMENDED ACTIONS")
    print("-" * 60)

    for index, action in enumerate(
        report["recommended_actions"],
        start=1
    ):
        print(f"{index}. {action}")

    print("\n" + "-" * 60)
    print("REPORT LIMITATIONS")
    print("-" * 60)

    for limitation in report["limitations"]:
        print(f"- {limitation}")

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
