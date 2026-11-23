
from pathlib import Path
import json

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"


class FraudEDA:

    def __init__(self, data):

        self.data = data.copy()
        self.target = "fraud"

        REPORT_DIR.mkdir(exist_ok=True)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)

        if self.target not in self.data.columns:
            raise ValueError("Fraud target column not found.")

    def run_all(self):

        print("\nStarting FraudLens Exploratory Data Analysis...")

        results = {}

        results["fraud_distribution"] = self.analyze_fraud_distribution()
        results["amount_analysis"] = self.analyze_amounts()
        results["category_analysis"] = self.analyze_categories()
        results["demographic_analysis"] = self.analyze_demographics()
        results["temporal_analysis"] = self.analyze_time()
        results["correlation_analysis"] = self.analyze_correlations()

        self.save_summary(results)

        print("\nEDA completed successfully.")
        print(f"Interactive charts saved in: {FIGURE_DIR.resolve()}")
        print(f"Summary saved in: {REPORT_DIR / 'eda_summary.json'}")

        return results

    # --------------------------------------------------
    # 1. FRAUD DISTRIBUTION
    # --------------------------------------------------

    def analyze_fraud_distribution(self):

        counts = self.data[self.target].value_counts().sort_index()

        fig = px.pie(
            values=counts.values,
            names=["Legitimate" if str(x) == "0" else "Fraud"
                   for x in counts.index],
            title="Fraud vs Legitimate Transactions",
            hole=0.45
        )

        fig.write_html(FIGURE_DIR / "fraud_distribution.html")

        fraud_count = int((self.data[self.target] == 1).sum())
        legitimate_count = int((self.data[self.target] == 0).sum())

        fraud_rate = fraud_count / len(self.data) * 100

        return {
            "fraud_count": fraud_count,
            "legitimate_count": legitimate_count,
            "fraud_percentage": round(fraud_rate, 4),
            "legitimate_percentage": round(100 - fraud_rate, 4)
        }

    # --------------------------------------------------
    # 2. AMOUNT ANALYSIS
    # --------------------------------------------------

    def analyze_amounts(self):

        fraud_data = self.data[self.data[self.target] == 1]
        legitimate_data = self.data[self.data[self.target] == 0]

        fig = px.box(
            self.data,
            x=self.target,
            y="amount",
            color=self.target,
            title="Transaction Amount Distribution: Fraud vs Legitimate",
            points="outliers"
        )

        fig.write_html(FIGURE_DIR / "amount_boxplot.html")

        fig2 = px.histogram(
            self.data,
            x="amount",
            color=self.target,
            nbins=50,
            barmode="overlay",
            opacity=0.65,
            title="Transaction Amount Histogram"
        )

        fig2.write_html(FIGURE_DIR / "amount_histogram.html")

        return {
            "fraud_amount": {
                "mean": round(float(fraud_data["amount"].mean()), 2),
                "median": round(float(fraud_data["amount"].median()), 2),
                "maximum": round(float(fraud_data["amount"].max()), 2)
            },
            "legitimate_amount": {
                "mean": round(float(legitimate_data["amount"].mean()), 2),
                "median": round(float(legitimate_data["amount"].median()), 2),
                "maximum": round(float(legitimate_data["amount"].max()), 2)
            }
        }

    # --------------------------------------------------
    # 3. CATEGORY ANALYSIS
    # --------------------------------------------------

    def analyze_categories(self):

        category_stats = (
            self.data.groupby("category")
            .agg(
                transaction_count=("fraud", "size"),
                fraud_count=("fraud", "sum"),
                fraud_rate=("fraud", "mean"),
                average_amount=("amount", "mean")
            )
            .reset_index()
        )

        category_stats["fraud_rate"] *= 100

        category_stats = category_stats.sort_values(
            "fraud_rate", ascending=False
        )

        fig = px.bar(
            category_stats,
            x="category",
            y="fraud_rate",
            color="fraud_rate",
            title="Fraud Rate by Transaction Category",
            labels={"fraud_rate": "Fraud Rate (%)"},
            hover_data=[
                "transaction_count",
                "fraud_count",
                "average_amount"
            ]
        )

        fig.update_layout(xaxis_tickangle=-45)

        fig.write_html(FIGURE_DIR / "category_fraud_rate.html")

        fig2 = px.bar(
            category_stats.sort_values("transaction_count", ascending=False),
            x="category",
            y="transaction_count",
            title="Transaction Volume by Category"
        )

        fig2.update_layout(xaxis_tickangle=-45)

        fig2.write_html(FIGURE_DIR / "category_volume.html")

        category_stats.to_csv(
            REPORT_DIR / "category_analysis.csv",
            index=False
        )

        return category_stats.round(4).to_dict(orient="records")

    # --------------------------------------------------
    # 4. DEMOGRAPHIC ANALYSIS
    # --------------------------------------------------

    def analyze_demographics(self):

        results = {}

        for column in ["age", "gender"]:

            if column not in self.data.columns:
                continue

            stats = (
                self.data.groupby(column)
                .agg(
                    transaction_count=("fraud", "size"),
                    fraud_count=("fraud", "sum"),
                    fraud_rate=("fraud", "mean"),
                    average_amount=("amount", "mean")
                )
                .reset_index()
            )

            stats["fraud_rate"] *= 100

            fig = px.bar(
                stats,
                x=column,
                y="fraud_rate",
                color="fraud_rate",
                title=f"Fraud Rate by {column.title()}",
                labels={"fraud_rate": "Fraud Rate (%)"},
                hover_data=[
                    "transaction_count",
                    "fraud_count",
                    "average_amount"
                ]
            )

            fig.write_html(
                FIGURE_DIR / f"fraud_by_{column}.html"
            )

            results[column] = stats.round(4).to_dict(orient="records")

        return results

    # --------------------------------------------------
    # 5. TEMPORAL ANALYSIS
    # --------------------------------------------------

    def analyze_time(self):

        if "step" not in self.data.columns:
            return {"warning": "Step column not available"}

        time_stats = (
            self.data.groupby("step")
            .agg(
                transaction_count=("fraud", "size"),
                fraud_count=("fraud", "sum"),
                fraud_rate=("fraud", "mean"),
                average_amount=("amount", "mean")
            )
            .reset_index()
        )

        time_stats["fraud_rate"] *= 100

        fig = px.line(
            time_stats,
            x="step",
            y="transaction_count",
            title="Transaction Volume Across Steps"
        )

        fig.write_html(FIGURE_DIR / "transaction_volume_over_time.html")

        fig2 = px.line(
            time_stats,
            x="step",
            y="fraud_count",
            title="Fraud Transactions Across Steps"
        )

        fig2.write_html(FIGURE_DIR / "fraud_count_over_time.html")

        fig3 = px.line(
            time_stats,
            x="step",
            y="fraud_rate",
            title="Fraud Rate Across Steps"
        )

        fig3.write_html(FIGURE_DIR / "fraud_rate_over_time.html")

        time_stats.to_csv(
            REPORT_DIR / "temporal_analysis.csv",
            index=False
        )

        return {
            "total_steps": int(self.data["step"].nunique()),
            "first_step": int(self.data["step"].min()),
            "last_step": int(self.data["step"].max()),
            "peak_transaction_step": int(
                time_stats.loc[
                    time_stats["transaction_count"].idxmax(), "step"
                ]
            ),
            "peak_fraud_step": int(
                time_stats.loc[
                    time_stats["fraud_count"].idxmax(), "step"
                ]
            )
        }

    # --------------------------------------------------
    # 6. CORRELATION ANALYSIS
    # --------------------------------------------------

    def analyze_correlations(self):

        numeric_data = self.data.select_dtypes(include=np.number)

        correlation = numeric_data.corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Numeric Feature Correlation Matrix"
        )

        fig.write_html(FIGURE_DIR / "correlation_matrix.html")

        return correlation.round(4).to_dict()

    # --------------------------------------------------
    # 7. SAVE REPORT
    # --------------------------------------------------

    def save_summary(self, results):

        output_path = REPORT_DIR / "eda_summary.json"

        with open(output_path, "w", encoding="utf-8") as file:

            json.dump(
                results,
                file,
                indent=4,
                default=str
            )

        print(f"EDA summary saved to {output_path}")