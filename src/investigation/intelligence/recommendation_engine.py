
from typing import Any, Dict, List


class InvestigationRecommendationEngine:
    """
    Generates investigation recommendations from transaction risk signals
    and model fraud probabilities.

    Recommendations are investigative suggestions, not automated actions.
    """

    SIGNAL_RECOMMENDATIONS = {
        "extreme_amount": {
            "priority": "high",
            "action": "Review the transaction amount against the customer's transaction history.",
            "reason": "The amount is at or above the reference 99th percentile.",
        },
        "high_amount": {
            "priority": "medium",
            "action": "Compare the transaction amount with typical transaction values.",
            "reason": "The amount is unusually high relative to the reference population.",
        },
        "rare_category": {
            "priority": "low",
            "action": "Review recent transactions involving this category.",
            "reason": "The transaction category occurs infrequently in reference data.",
        },
        "unknown_category": {
            "priority": "medium",
            "action": "Validate the transaction category and check category mapping.",
            "reason": "The category was not found in the reference data.",
        },
        "missing_category": {
            "priority": "medium",
            "action": "Verify and complete the missing transaction category.",
            "reason": "Transaction category information is unavailable.",
        },
        "invalid_amount": {
            "priority": "high",
            "action": "Validate the transaction amount before completing the investigation.",
            "reason": "The amount is missing or cannot be interpreted numerically.",
        },
        "negative_amount": {
            "priority": "high",
            "action": "Verify whether the negative amount represents a valid reversal or a data error.",
            "reason": "The transaction amount is negative.",
        },
        "missing_age": {
            "priority": "low",
            "action": "Check whether the missing age field can be validated from an authorized source.",
            "reason": "Age information is missing from the transaction record.",
        },
        "missing_gender": {
            "priority": "low",
            "action": "Validate the missing demographic field if it is required for data quality.",
            "reason": "Gender information is missing from the transaction record.",
        },
    }

    def generate_recommendations(
        self,
        signals: List[Dict[str, Any]],
        fraud_probability: float = 0.0,
    ) -> List[Dict[str, Any]]:

        recommendations = []
        seen_actions = set()

        # --------------------------------------------------
        # 1. Recommendations from detected risk signals
        # --------------------------------------------------

        for signal in signals:

            signal_name = signal.get("signal")

            recommendation_data = self.SIGNAL_RECOMMENDATIONS.get(
                signal_name
            )

            if recommendation_data is None:
                continue

            action = recommendation_data["action"]

            # Avoid duplicate recommendations
            if action in seen_actions:
                continue

            seen_actions.add(action)

            recommendations.append({
                "priority": recommendation_data["priority"],
                "action": action,
                "reason": recommendation_data["reason"],
                "source_signal": signal_name,
            })

        # --------------------------------------------------
        # 2. Model probability recommendations
        # --------------------------------------------------

        if fraud_probability >= 0.90:

            action = (
                "Prioritize this transaction for immediate manual review."
            )

            if action not in seen_actions:
                recommendations.append({
                    "priority": "high",
                    "action": action,
                    "reason": (
                        "The model assigned a fraud probability "
                        "of 90% or higher."
                    ),
                    "source_signal": "model_probability",
                })

        elif fraud_probability >= 0.70:

            action = (
                "Conduct a detailed review before closing this case."
            )

            if action not in seen_actions:
                recommendations.append({
                    "priority": "high",
                    "action": action,
                    "reason": (
                        "The model assigned a fraud probability "
                        "between 70% and 90%."
                    ),
                    "source_signal": "model_probability",
                })

        elif fraud_probability >= 0.40:

            action = (
                "Review the flagged indicators and gather additional evidence."
            )

            if action not in seen_actions:
                recommendations.append({
                    "priority": "medium",
                    "action": action,
                    "reason": (
                        "The model assigned a moderate fraud probability."
                    ),
                    "source_signal": "model_probability",
                })

        # --------------------------------------------------
        # 3. Sort by investigation priority
        # --------------------------------------------------

        priority_order = {
            "high": 1,
            "medium": 2,
            "low": 3,
        }

        recommendations.sort(
            key=lambda item: priority_order.get(
                item["priority"], 99
            )
        )

        return recommendations
