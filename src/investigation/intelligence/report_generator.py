
from pathlib import Path
from html import escape


class InvestigationReportGenerator:

    def generate_html(self, report):
        metadata = report.get("report_metadata", {})
        case_info = report.get("case_information", {})
        risk = report.get("risk_assessment", {})

        evidence = report.get("observed_evidence", [])
        shap = report.get("shap_explanations", [])
        reasoning = report.get("reasoning", [])
        actions = report.get("recommended_actions", [])
        limitations = report.get("limitations", [])

        risk_tier = str(risk.get("tier", "UNKNOWN")).upper()
        probability = float(risk.get("probability", 0))
        risk_score = risk.get("risk_score", 0)

        tier_colors = {
            "CRITICAL": "#ef4444",
            "HIGH": "#f97316",
            "MEDIUM": "#eab308",
            "LOW": "#22c55e",
        }

        tier_color = tier_colors.get(risk_tier, "#64748b")

        evidence_rows = ""

        for item in evidence:
            evidence_rows += f"""
            <tr>
                <td>{escape(str(item.get('description', '')))}</td>
                <td>{escape(str(item.get('value', '')))}</td>
            </tr>
            """

        shap_rows = ""

        for item in shap:
            contribution = float(item.get("contribution", 0))

            contribution_color = (
                "#ef4444" if contribution > 0 else "#22c55e"
            )

            shap_rows += f"""
            <tr>
                <td>{escape(str(item.get('feature', '')))}</td>
                <td style="color:{contribution_color};font-weight:bold;">
                    {contribution:+.6f}
                </td>
                <td>{escape(str(item.get('impact', '')))}</td>
            </tr>
            """

        reasoning_html = "".join(
            f"<li>{escape(str(item))}</li>"
            for item in reasoning
        )

        actions_html = "".join(
            f"<li>{escape(str(item))}</li>"
            for item in actions
        )

        limitations_html = "".join(
            f"<li>{escape(str(item))}</li>"
            for item in limitations
        )

        shap_section = ""

        if shap:
            shap_section = f"""
            <section class="card">
                <h2>Model Explainability — SHAP</h2>
                <p class="muted">
                    Positive values increase the model's fraud prediction;
                    negative values decrease it. SHAP values describe
                    model behavior, not proof of fraudulent activity.
                </p>

                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Contribution</th>
                            <th>Impact</th>
                        </tr>
                    </thead>
                    <tbody>
                        {shap_rows}
                    </tbody>
                </table>
            </section>
            """

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>FraudLens Investigation Report</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #0b1120;
    color: #e5e7eb;
    font-family: Arial, Helvetica, sans-serif;
    padding: 35px;
}}

.container {{
    max-width: 1050px;
    margin: auto;
}}

.header {{
    padding: 30px;
    background: #111827;
    border: 1px solid #273449;
    border-radius: 14px;
    margin-bottom: 24px;
}}

.brand {{
    color: #38bdf8;
    letter-spacing: 2px;
    font-weight: bold;
    font-size: 13px;
}}

.header h1 {{
    margin: 12px 0;
    font-size: 30px;
}}

.muted {{
    color: #9ca3af;
    line-height: 1.6;
}}

.metrics {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}}

.metric {{
    background: #111827;
    border: 1px solid #273449;
    border-radius: 12px;
    padding: 22px;
}}

.metric-label {{
    color: #9ca3af;
    font-size: 13px;
}}

.metric-value {{
    font-size: 27px;
    font-weight: bold;
    margin-top: 12px;
}}

.card {{
    background: #111827;
    border: 1px solid #273449;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}}

h2 {{
    font-size: 19px;
    margin-top: 0;
    margin-bottom: 18px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th, td {{
    padding: 13px;
    text-align: left;
    border-bottom: 1px solid #273449;
}}

th {{
    color: #9ca3af;
    font-size: 13px;
}}

li {{
    margin-bottom: 12px;
    line-height: 1.6;
}}

.risk-badge {{
    display: inline-block;
    padding: 8px 14px;
    border-radius: 8px;
    color: white;
    font-weight: bold;
    background: {tier_color};
}}

.footer {{
    padding: 20px;
    text-align: center;
    color: #6b7280;
    font-size: 12px;
}}

@media (max-width: 700px) {{
    body {{
        padding: 12px;
    }}

    .metrics {{
        grid-template-columns: 1fr;
    }}

    table {{
        font-size: 12px;
    }}
}}

@media print {{
    body {{
        background: white;
        color: black;
        padding: 0;
    }}

    .header, .card, .metric {{
        background: white;
        color: black;
        border: 1px solid #ddd;
    }}

    .muted, .metric-label, th {{
        color: #555;
    }}

    .container {{
        max-width: 100%;
    }}
}}

</style>
</head>

<body>

<div class="container">

<div class="header">
    <div class="brand">FRAUDLENS AI</div>
    <h1>Transaction Investigation Report</h1>

    <p>Case ID: <strong>{escape(str(case_info.get('case_id', 'N/A')))}</strong></p>

    <p>
        Status:
        <strong>{escape(str(case_info.get('status', 'N/A')))}</strong>
    </p>

    <p class="muted">
        Generated: {escape(str(metadata.get('generated_at', 'N/A')))}
    </p>
</div>

<div class="metrics">

    <div class="metric">
        <div class="metric-label">Fraud Probability</div>
        <div class="metric-value">{probability:.2%}</div>
    </div>

    <div class="metric">
        <div class="metric-label">Risk Score</div>
        <div class="metric-value">{escape(str(risk_score))}/100</div>
    </div>

    <div class="metric">
        <div class="metric-label">Risk Classification</div>
        <div class="metric-value">
            <span class="risk-badge">{escape(risk_tier)}</span>
        </div>
    </div>

</div>

<section class="card">
    <h2>Observed Transaction Evidence</h2>

    <table>
        <thead>
            <tr>
                <th>Evidence Field</th>
                <th>Observed Value</th>
            </tr>
        </thead>

        <tbody>
            {evidence_rows}
        </tbody>
    </table>
</section>

<section class="card">
    <h2>Investigation Reasoning</h2>
    <ul>
        {reasoning_html}
    </ul>
</section>

{shap_section}

<section class="card">
    <h2>Recommended Analyst Actions</h2>
    <ol>
        {actions_html}
    </ol>
</section>

<section class="card">
    <h2>Limitations & Disclaimer</h2>
    <ul>
        {limitations_html}
    </ul>

    <p class="muted">
        This report is decision-support material. A model prediction does not
        independently establish fraud. Final determinations must be made by
        authorized personnel following verification of available evidence.
    </p>
</section>

<div class="footer">
    FraudLens AI &bull; Explainable Transaction Risk Intelligence
    <br>
    Engine Version: {escape(str(metadata.get('engine_version', 'N/A')))}
</div>

</div>
</body>
</html>
"""

    def save_html(self, html_content, output_path):
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path.write_text(
            html_content,
            encoding="utf-8"
        )

        return output_path
