from pathlib import Path
import json
import pandas as pd

from src.investigation.intelligence.report_generator import (
    InvestigationReportGenerator,
)


def main():

    print("=" * 60)
    print("FRAUDLENS AI - SHAP INTEGRATED REPORT")
    print("=" * 60)

    case_id = "FL-001315"

    # --------------------------------------------------
    # 1. LOAD INVESTIGATION REPORT
    # --------------------------------------------------

    report_path = Path(
        f"reports/investigations/{case_id}_investigation.json"
    )

    if not report_path.exists():
        print("Investigation JSON report not found.")
        print("Run the investigation intelligence script first.")
        return

    with open(report_path, "r", encoding="utf-8") as file:
        investigation_report = json.load(file)

    # --------------------------------------------------
    # 2. LOAD CASE-SPECIFIC SHAP EXPLANATIONS
    # --------------------------------------------------

    shap_path = Path(
        f"reports/shap/cases/{case_id}/case_shap_explanation.csv"
    )

    if not shap_path.exists():
        print("Case-specific SHAP explanation not found.")
        print("Run explain_investigation_case.py first.")
        return

    shap_df = pd.read_csv(shap_path)

    # --------------------------------------------------
    # 3. CONVERT SHAP DATA TO REPORT FORMAT
    # --------------------------------------------------

    shap_explanations = []

    for _, row in shap_df.iterrows():

        contribution = float(row["shap_value"])

        if contribution > 0:
            impact = "Increases predicted fraud probability"
        elif contribution < 0:
            impact = "Decreases predicted fraud probability"
        else:
            impact = "Neutral contribution"

        shap_explanations.append({
            "feature": str(row["feature"]),
            "contribution": contribution,
            "impact": impact
            })

    # --------------------------------------------------
    # 4. INJECT SHAP INTO INVESTIGATION REPORT
    # --------------------------------------------------

    investigation_report["shap_explanations"] = (
        shap_explanations
    )

    print(f"\nCase ID: {case_id}")
    print(f"SHAP features loaded: {len(shap_explanations)}")

    # --------------------------------------------------
    # 5. GENERATE HTML REPORT
    # --------------------------------------------------

    generator = InvestigationReportGenerator()

    html_content = generator.generate_html(
        investigation_report
    )

    output_path = Path(
        f"reports/investigations/{case_id}_investigation.html"
    )

    generator.save_html(
        html_content,
        output_path
    )

    # --------------------------------------------------
    # 6. OPEN REPORT
    # --------------------------------------------------

    print("\nHTML report generated successfully!")
    print(f"Report location: {output_path.resolve()}")

    import webbrowser

    webbrowser.open(
        output_path.resolve().as_uri()
    )

    print("\nReport opened in browser.")
    print("=" * 60)


if __name__ == "__main__":
    main()