
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)

from src.investigation.case_manager import CaseManager
from src.investigation.intelligence.investigation_engine import InvestigationIntelligenceEngine
from src.investigation.intelligence.report_generator import InvestigationReportGenerator
from src.investigation.intelligence.pdf_generator import InvestigationPDFGenerator


# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

manager = CaseManager()
report_engine = InvestigationIntelligenceEngine()
html_report_engine = InvestigationReportGenerator()
pdf_report_engine = InvestigationPDFGenerator()


# ==========================================================
# DESIGN SYSTEM
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b1120;
        color: #e6edf7;
    }

    [data-testid="stSidebar"] {
        background: #0f1728;
        border-right: 1px solid #202d43;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #151f32, #111a2b);
        border: 1px solid #263650;
        padding: 20px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700;
    }

    h1, h2, h3 {
        color: #f1f5f9 !important;
    }

    p, label, span {
        color: inherit;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #263650;
        border-radius: 12px;
        overflow: hidden;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .hero-card {
        background: linear-gradient(120deg, #16243a, #111a2c);
        border: 1px solid #273952;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 14px;
    }

    .hero-sub {
        color: #94a3b8;
        font-size: 14px;
    }

    .risk-critical {
        color: #fb7185;
        font-weight: 700;
    }

    .risk-high {
        color: #fb923c;
        font-weight: 700;
    }

    .risk-medium {
        color: #facc15;
        font-weight: 700;
    }

    .risk-low {
        color: #34d399;
        font-weight: 700;
    }

    .small-note {
        color: #94a3b8;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# HELPERS
# ==========================================================

def money(value):
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def risk_color(tier):

    colors = {
        "CRITICAL": "#ef4444",
        "HIGH": "#f97316",
        "MEDIUM": "#eab308",
        "LOW": "#10b981"
    }

    return colors.get(tier, "#64748b")


def get_cases():

    data = manager.get_cases()

    if data.empty:
        return data

    if "fraud_probability" in data.columns:
        data["fraud_probability"] = pd.to_numeric(
            data["fraud_probability"],
            errors="coerce"
        ).fillna(0)

    if "risk_score" in data.columns:
        data["risk_score"] = pd.to_numeric(
            data["risk_score"],
            errors="coerce"
        ).fillna(0)

    if "amount" in data.columns:
        data["amount"] = pd.to_numeric(
            data["amount"],
            errors="coerce"
        ).fillna(0)

    return data


def build_risk_donut(data):

    tier_order = [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW"
    ]

    counts = (
        data["risk_tier"]
        .value_counts()
        .reindex(tier_order, fill_value=0)
    )

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.72,
        color=counts.index,
        color_discrete_map={
            "CRITICAL": "#ef4444",
            "HIGH": "#f97316",
            " MEDIUM": "#eab308",
            "LOW": "#10b981"
        }
    )

    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        marker=dict(
            line=dict(
                color="#111827",
                width=3
            )
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        legend=dict(
            orientation="h",
            y=-0.12
        ),
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=35
        ),
        height=340
    )

    return fig


def build_risk_bar(data):

    tier_order = [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW"
    ]

    counts = (
        data["risk_tier"]
        .value_counts()
        .reindex(tier_order, fill_value=0)
        .reset_index()
    )

    counts.columns = [
        "Risk Tier",
        "Transactions"
    ]

    fig = px.bar(
        counts,
        x="Risk Tier",
        y="Transactions",
        color="Risk Tier",
        color_discrete_map={
            "CRITICAL": "#ef4444",
            " HIGH": "#f97316",
            "MEDIUM": "#eab308",
            "LOW": "#10b981"
        },
        text="Transactions"
    )

    fig.update_traces(
        textposition="outside",
        marker_line_width=0
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        showlegend=False,
        margin=dict(
            l=10,
            r=10,
            t=25,
            b=10
        ),
        height=330,
        xaxis=dict(
            title="",
            showgrid=False
        ),
        yaxis=dict(
            title="Transactions",
            gridcolor="#263449"
        )
    )

    return fig


def styled_dataframe(data):

    return data.style.format({
        "amount": "{:,.2f}",
        "fraud_probability": "{:.2%}",
        "risk_score": "{:.0f}"
    }, na_rep="—")


# ==========================================================
# DATA
# ==========================================================

cases = get_cases()

if cases.empty:

    st.title("🛡️ FraudLens AI")

    st.warning(
        "No investigation cases found in the database."
    )

    if st.button("Import risk report"):

        try:

            imported = manager.import_transactions()

            st.success(
                f"Imported {imported} transactions."
            )

            st.rerun()

        except Exception as error:

            st.error(str(error))

    st.stop()


total_transactions = len(cases)

flagged_transactions = int(
    cases["review_flag"].sum()
)

critical_count = int(
    (cases["risk_tier"] == "CRITICAL").sum()
)

high_count = int(
    (cases["risk_tier"] == "HIGH").sum()
)

medium_count = int(
    (cases["risk_tier"] == "MEDIUM").sum()
)

low_count = int(
    (cases["risk_tier"] == "LOW").sum()
)

pending_count = int(
    (cases["status"] == "Pending").sum()
)

confirmed_count = int(
    (cases["status"] == "Confirmed Fraud").sum()
)

investigating_count = int(
    (cases["status"] == "Investigating").sum()
)

legitimate_count = int(
    (cases["status"] == "Legitimate").sum()
)

avg_score = float(
    cases["risk_score"].mean()
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("# 🛡️ FraudLens")
    st.caption("TRANSACTION RISK INTELLIGENCE")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Investigation Queue",
            "Case Workspace"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### Filters")

    filter_risk = st.multiselect(
        "Risk level",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH"]
    )

    filter_status = st.multiselect(
        "Investigation status",
        [
            "Pending",
            "Investigating",
            "Confirmed Fraud",
            "Legitimate"
        ],
        default=[
            "Pending",
            "Investigating"
        ]
    )

    flagged_only = st.checkbox(
        "Review flagged transactions only",
        value=True
    )

    st.markdown("---")

    st.markdown("### System status")

    st.markdown(
        """
        <div style="color:#34d399;font-weight:600">
        ● Risk engine operational
        </div>
        <div style="color:#94a3b8;font-size:13px">
        Investigation database connected
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.caption("FraudLens AI · Research prototype")


# ==========================================================
# FILTERED DATA
# ==========================================================

filtered = cases.copy()

if filter_risk:
    filtered = filtered[
        filtered["risk_tier"].isin(filter_risk)
    ]
else:
    filtered = filtered.iloc[0:0]

if filter_status:
    filtered = filtered[
        filtered["status"].isin(filter_status)
    ]
else:
    filtered = filtered.iloc[0:0]

if flagged_only:
    filtered = filtered[
        filtered["review_flag"] == 1
    ]

filtered = filtered.sort_values(
    "risk_score",
    ascending=False
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin:0">Fraud Intelligence Platform</h1>
        <div class="hero-sub">
            AI-powered transaction monitoring, risk analysis,
            and investigation management.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# KPI CARDS
# ==========================================================

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

m2.metric(
    "Flagged",
    f"{flagged_transactions:,}",
    f"{flagged_transactions / max(total_transactions, 1) * 100:.2f}% of volume"
)

m3.metric(
    "Critical Risk",
    f"{critical_count:,}"
)

m4.metric(
    "Under Investigation",
    f"{investigating_count:,}"
)

m5.metric(
    "Average Risk Score",
    f"{avg_score:.1f}/100"
)


# ==========================================================
# NAVIGATION CONTENT
# ==========================================================

if page == "Overview":

    st.markdown("## Risk Overview")

    left, right = st.columns(
        [1.05, 1.6]
    )

    with left:

        st.markdown("### Risk Composition")

        st.plotly_chart(
            build_risk_donut(cases),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with right:

        st.markdown("### Risk Tier Distribution")

        st.plotly_chart(
            build_risk_bar(cases),
            use_container_width=True,
            config={"displayModeBar": False}
        )


    # Status metrics

    st.markdown("## Investigation Status")

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Pending",
        f"{pending_count:,}"
    )

    s2.metric(
        "Investigating",
        f"{investigating_count:,}"
    )

    s3.metric(
        "Confirmed Fraud",
        f"{confirmed_count:,}"
    )

    s4.metric(
        "Cleared / Legitimate",
        f"{legitimate_count:,}"
    )


    # Priority transactions

    st.markdown("## Priority Alerts")

    priority = cases[
        cases["review_flag"] == 1
    ].sort_values(
        "risk_score",
        ascending=False
    ).head(10)

    if not priority.empty:

        priority_columns = [
            "case_id",
            "transaction_id",
            "amount",
            "category",
            "risk_score",
            "risk_tier",
            "status"
        ]

        priority_columns = [
            col for col in priority_columns
            if col in priority.columns
        ]

        st.dataframe(
            priority[priority_columns],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No flagged transactions.")


elif page == "Investigation Queue":

    st.markdown("## Investigation Queue")

    st.caption(
        "Filtered, prioritized transaction cases."
    )

    search = st.text_input(
        "Search by case ID, transaction ID, or category"
    )

    queue = filtered.copy()

    if search:

        search_lower = search.lower()

        search_cols = [
            col for col in [
                "case_id",
                "transaction_id",
                "category"
            ]
            if col in queue.columns
        ]

        if search_cols:

            mask = queue[
                search_cols
            ].astype(str).apply(
                lambda row: row.str.lower().str.contains(
                    search_lower,
                    regex=False
                ).any(),
                axis=1
            )

            queue = queue[mask]


    st.markdown(
        f"**{len(queue):,} transactions** match the selected filters."
    )

    if not queue.empty:

        queue_columns = [
            col for col in [
                "case_id",
                "transaction_id",
                "amount",
                "category",
                "fraud_probability",
                "risk_score",
                "risk_tier",
                "status",
                "recommended_action"
            ]
            if col in queue.columns
        ]

        queue_display = queue[
            queue_columns
        ].copy()

        if "fraud_probability" in queue_display.columns:

            queue_display["fraud_probability"] = (
                queue_display["fraud_probability"] * 100
            ).round(2)

            queue_display = queue_display.rename(
                columns={
                    "fraud_probability": "Fraud probability (%)"
                }
            )

        st.dataframe(
            queue_display,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download filtered queue",
            data=queue.to_csv(index=False),
            file_name="fraudlens_filtered_queue.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No transactions match your current filters. "
            "Adjust filters in the sidebar."
        )


elif page == "Case Workspace":

    st.markdown("## Case Workspace")

    st.caption(
        "Inspect a transaction and update its investigation record."
    )

    available_cases = cases.sort_values(
        "risk_score",
        ascending=False
    )

    selected_id = st.selectbox(
        "Select investigation case",
        available_cases["case_id"].tolist()
    )

    selected = manager.get_case(
        selected_id
    )

    if selected:

        transaction = selected["transaction_data"]

        st.markdown(
            f"### Case {selected['case_id']}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Risk Score",
            f"{selected['risk_score']}/100"
        )

        c2.metric(
            "Risk Tier",
            selected["risk_tier"]
        )

        c3.metric(
            "Fraud Probability",
            f"{selected['fraud_probability'] * 100:.2f}%"
        )

        c4.metric(
            "Current Status",
            selected["status"]
        )

        st.markdown("---")

        detail_left, detail_right = st.columns(2)

        with detail_left:

            st.markdown("### Transaction Information")

            st.write(
                "**Transaction ID:**",
                selected["transaction_id"]
            )

            st.write(
                "**Amount:**",
                money(transaction.get("amount"))
            )

            st.write(
                "**Category:**",
                transaction.get("category", "N/A")
            )

            st.write(
                "**Age:**",
                transaction.get("age", "N/A")
            )

            st.write(
                "**Gender:**",
                transaction.get("gender", "N/A")
            )

            st.write(
                "**Step:**",
                transaction.get("step", "N/A")
            )

            st.write(
                "**Recommended Action:**",
                selected.get(
                    "recommended_action",
                    "N/A"
                )
            )

        with detail_right:

            st.markdown("### Risk Assessment")

            risk_value = selected["fraud_probability"]

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_value * 100,
                    number={
                        "suffix": "%",
                        "font": {
                            "color": "#e2e8f0",
                            "size": 36
                        }
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickcolor": "#94a3b8"
                        },
                        "bar": {
                            "color": risk_color(
                                selected["risk_tier"]
                            )
                        },
                        "bgcolor": "#172235",
                        "borderwidth": 0,
                        "steps": [
                            {
                                "range": [0, 30],
                                "color": "#12352d"
                            },
                            {
                                "range": [30, 60],
                                "color": "#3b341b"
                            },
                            {
                                "range": [60, 85],
                                "color": "#402b1b"
                            },
                            {
                                "range": [85, 100],
                                "color": "#421f29"
                            }
                        ]
                    }
                )
            )

            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=280,
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=10
                )
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )


        # Status management

        st.markdown("---")
        st.markdown("### Update Case")

        status_options = [
            "Pending",
            "Investigating",
            "Confirmed Fraud",
            "Legitimate"
        ]

        new_status = st.selectbox(
            "Investigation status",
            status_options,
            index=status_options.index(
                selected["status"]
            ),
            key=f"workspace_status_{selected_id}"
        )

        status_note = st.text_input(
            "Status change reason",
            key=f"workspace_reason_{selected_id}",
            placeholder="Optional explanation for the status change"
        )

        if st.button(
            "Update Case Status",
            type="primary"
        ):

            manager.update_status(
                selected_id,
                new_status,
                status_note
            )

            st.success("Status updated successfully.")
            st.rerun()


        # Notes

        st.markdown("### Analyst Notes")

        existing_notes = selected.get(
            "analyst_notes",
            ""
        )

        if existing_notes:

            st.text_area(
                "Previous notes",
                existing_notes,
                height=160,
                disabled=True
            )

        note = st.text_area(
            "Add new note",
            height=120,
            key=f"workspace_note_{selected_id}",
            placeholder="Enter investigation findings..."
        )

        if st.button(
            "Save Investigation Note"
        ):

            if note.strip():

                manager.add_note(
                    selected_id,
                    note
                )

                st.success("Note saved.")
                st.rerun()

            else:

                st.warning(
                    "Enter a note before saving."
                )


        # History

        st.markdown("---")
        st.markdown("### Case Activity History")

        history = manager.get_history(
            selected_id
        )

        if not history.empty:

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.caption(
                "No activity recorded for this case yet."
            )
        # ==========================================================
        # SHAP AI EXPLAINABILITY
        # ==========================================================

        st.markdown("---")
        st.markdown("## 🧠 AI Risk Explanation")

        st.caption(
            "Understand which features influenced the model's fraud prediction."
        )

        shap_file = (
            f"reports/shap/cases/{selected_id}/"
            "case_shap_explanation.csv"
        )

        if os.path.exists(shap_file):

            shap_data = pd.read_csv(shap_file)

            shap_data = shap_data.sort_values(
                "shap_value",
                key=lambda x: x.abs(),
                ascending=True
            )

            shap_data["color"] = shap_data["shap_value"].apply(
                lambda x: "#ef4444" if x > 0 else "#10b981"
            )

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=shap_data["shap_value"],
                    y=shap_data["feature"],
                    orientation="h",
                    marker_color=shap_data["color"],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "SHAP Contribution: %{x:.5f}"
                        "<extra></extra>"
                    )
                )
            )

            fig.update_layout(
                title="Feature Contribution to Fraud Prediction",
                xaxis_title="SHAP Value",
                yaxis_title="Feature",
                template="plotly_dark",
                height=650,
                showlegend=False,
                margin=dict(l=20, r=20, t=60, b=40)
            )

            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="white",
                opacity=0.5
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            risk_increasing = shap_data[
                shap_data["shap_value"] > 0
            ]

            risk_decreasing = shap_data[
                shap_data["shap_value"] < 0
            ]

            col1, col2 = st.columns(2)

            col1.metric(
                "Risk-Increasing Factors",
                len(risk_increasing)
            )

            col2.metric(
                "Risk-Reducing Factors",
                len(risk_decreasing)
            )

            st.markdown("### Feature Contribution Details")

            display_data = shap_data.sort_values(
                "absolute_impact",
                ascending=False
            )

            st.dataframe(
                display_data[
                    [
                        "feature",
                        "feature_value",
                        "shap_value",
                        "impact"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "SHAP explanation is not yet available for this case."
            )

        st.caption(
            "SHAP values describe model behavior, not proof of fraud. "
            "All findings must be independently verified."
        )
        # ============================================================
        # INVESTIGATION REPORT CENTER
        # ============================================================

        st.markdown("---")
        st.markdown("## 📑 Investigation Report Center")

        st.caption(
            "Generate downloadable investigation reports "
            "for the currently selected case."
        )

        st.info(
            "Reports combine case information, risk assessment, "
            "investigation evidence, SHAP explanations, and "
            "recommended analyst actions."
        )

        generate_report_btn = st.button(
            "📝 Generate Investigation Report",
            key="generate_investigation_report",
            use_container_width=True
        )
        if generate_report_btn:
            with st.spinner("Generating investigation report..."):

                # Load case-specific SHAP evidence
                report_shap_data = None

                report_shap_file = (
                    f"reports/shap/cases/{selected_id}/"
                    "case_shap_explanation.csv"
                )

                if os.path.exists(report_shap_file):
                    report_shap_data = pd.read_csv(report_shap_file)

                # Generate structured investigation report
                investigation_report = report_engine.generate_report(
                    selected,
                    shap_evidence=report_shap_data
                )

                st.session_state["investigation_report"] = investigation_report
                st.session_state["investigation_report_case"] = selected_id

                st.success("Investigation report generated successfully!")
                # ============================================================
        # REPORT PREVIEW
        # ============================================================

        if (
            "investigation_report" in st.session_state
            and st.session_state.get("investigation_report_case") == selected_id
        ):
            st.markdown("---")
            st.markdown("### 📋 Generated Report Preview")

            st.json(
                st.session_state["investigation_report"],
                expanded=1
            )
                        # JSON REPORT DOWNLOAD
            import json

            json_report = json.dumps(
                st.session_state["investigation_report"],
                indent=4,
                default=str
            )

            st.download_button(
                label="⬇️ Download Investigation Report (JSON)",
                data=json_report,
                file_name=f"FraudLens_Report_{selected_id}.json",
                mime="application/json",
                key="download_investigation_json",
                use_container_width=True
            )
            # ============================================================
            # HTML REPORT DOWNLOAD
            # ============================================================

            html_report = html_report_engine.generate_html(
                st.session_state["investigation_report"]
            )

            st.download_button(
                label="🌐 Download Investigation Report (HTML)",
                data=html_report,
                file_name=f"FraudLens_Report_{selected_id}.html",
                mime="text/html",
                key="download_investigation_html",
                use_container_width=True
            )
                        # ============================================================
            # PDF REPORT DOWNLOAD
            # ============================================================

            pdf_bytes = pdf_report_engine.generate_pdf(
                st.session_state["investigation_report"]
            )

            st.download_button(
                label="📄 Download Investigation Report (PDF)",
                data=pdf_bytes,
                file_name=f"FraudLens_Report_{selected_id}.pdf",
                mime="application/pdf",
                key="download_investigation_pdf",
                use_container_width=True
            )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748b;font-size:12px">
        FraudLens AI · Explainable Financial Risk Intelligence
        <br>
        Machine-generated risk scores are indicators for investigation,
        not proof of fraud.
    </div>
    """,
    unsafe_allow_html=True
)
