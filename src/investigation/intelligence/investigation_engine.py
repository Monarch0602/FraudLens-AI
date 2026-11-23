
from datetime import datetime
from pathlib import Path
import json


class InvestigationIntelligenceEngine:
    """
    Generates evidence-grounded investigation intelligence.

    Separates:
    - Observed transaction data
    - ML risk inference
    - SHAP explanations
    - Analyst recommendations
    """

    def __init__(self):
        self.version = "1.0"

    def _get_transaction_data(self, case):
        return case.get("transaction_data", {}) or {}

    def _get_risk_level(self, case):
        tier = str(case.get("risk_tier", "UNKNOWN")).upper()
        probability = case.get("fraud_probability", 0)

        return {
            "tier": tier,
            "probability": float(probability),
            "risk_score": case.get("risk_score", 0),
            "review_flag": case.get("review_flag", False),
            "recommended_action": case.get(
                "recommended_action", "Manual review"
            ),
        }

    def _extract_observed_evidence(self, transaction):
        evidence = []

        fields = {
            "amount": "Transaction amount",
            "category": "Transaction category",
            "step": "Transaction time step",
            "age": "Customer age group",
            "gender": "Recorded gender category",
        }

        for field, description in fields.items():
            if field in transaction:
                evidence.append({
                    "type": "OBSERVED_DATA",
                    "field": field,
                    "description": description,
                    "value": transaction[field],
                })

        return evidence

    def _extract_shap_evidence(self, shap_evidence):
        if shap_evidence is None:
            return []

        evidence = []

        for _, row in shap_evidence.iterrows():
            shap_value = float(row["shap_value"])

            evidence.append({
                "type": "SHAP_CONTRIBUTION",
                "feature": str(row["feature"]),
                "contribution": shap_value,
                "impact": (
                    "Increased model fraud prediction"
                    if shap_value > 0
                    else "Decreased model fraud prediction"
                ),
            })

        evidence.sort(
            key=lambda x: abs(x["contribution"]),
            reverse=True
        )

        return evidence

    def _generate_reasoning(self, risk, observed, shap):
        reasoning = []

        probability = risk["probability"]
        tier = risk["tier"]

        reasoning.append(
            f"The model assigned a fraud probability of "
            f"{probability:.2%} and classified the transaction "
            f"as {tier} risk."
        )

        if observed:
            amount = next(
                (
                    item["value"]
                    for item in observed
                    if item["field"] == "amount"
                ),
                None,
            )

            category = next(
                (
                    item["value"]
                    for item in observed
                    if item["field"] == "category"
                ),
                None,
            )

            if amount is not None:
                reasoning.append(
                    f"The observed transaction amount is {amount}."
                )

            if category is not None:
                reasoning.append(
                    f"The recorded transaction category is {category}."
                )

        if shap:
            positive = [
                item for item in shap
                if item["contribution"] > 0
            ]

            if positive:
                top_features = positive[:3]

                feature_names = [
                    item["feature"]
                    for item in top_features
                ]

                reasoning.append(
                    "The model explanation indicates that the following "
                    "features contributed toward a higher fraud prediction: "
                    + ", ".join(feature_names)
                    + ". These are model signals, not proof of fraud."
                )

        return reasoning

    def _recommend_actions(self, risk):
        tier = risk["tier"]

        actions = {
            "CRITICAL": [
                "Prioritize this case for immediate analyst review.",
                "Verify the transaction against available authorized records.",
                "Check whether the transaction can be confirmed by the customer.",
                "Document the investigation outcome and supporting evidence.",
            ],
            "HIGH": [
                "Conduct a priority manual review.",
                "Validate the transaction details using available records.",
                "Review relevant account activity if authorized.",
                "Document findings before making a final decision.",
            ],
            "MEDIUM": [
                "Review transaction details for inconsistencies.",
                "Compare with available historical transaction patterns.",
                "Monitor for additional suspicious activity.",
            ],
            "LOW": [
                "Retain the transaction for routine monitoring.",
                "Escalate if additional risk signals emerge.",
            ],
        }

        return actions.get(
            tier,
            [
                "Manually review available transaction information.",
                "Verify relevant records before making a decision.",
            ],
        )

    def generate_report(self, case, shap_evidence=None):
        transaction = self._get_transaction_data(case)

        risk = self._get_risk_level(case)

        observed = self._extract_observed_evidence(transaction)

        shap = self._extract_shap_evidence(shap_evidence)

        reasoning = self._generate_reasoning(
            risk,
            observed,
            shap,
        )

        actions = self._recommend_actions(risk)

        report = {
            "report_metadata": {
                "engine_version": self.version,
                "generated_at": datetime.now().isoformat(),
                "case_id": case.get("case_id", "UNKNOWN"),
            },
            "case_information": {
                "case_id": case.get("case_id"),
                "status": case.get("status"),
            },
            "risk_assessment": risk,
            "observed_evidence": observed,
            "shap_explanations": shap,
            "reasoning": reasoning,
            "recommended_actions": actions,
            "limitations": [
                "A fraud probability is a model estimate, not a confirmed finding.",
                "SHAP values explain model behavior and do not establish causation.",
                "Only available transaction data is included in this report.",
                "Recommended actions require analyst verification.",
            ],
        }

        return report

    def save_report(self, report, output_path):
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                report,
                file,
                indent=4,
                default=str
            )

        return output_path
