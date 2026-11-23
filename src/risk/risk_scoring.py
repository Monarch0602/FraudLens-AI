
import numpy as np
import pandas as pd


class RiskScoringEngine:

    def __init__(self, review_threshold=0.85):

        self.review_threshold = review_threshold

    def calculate_score(self, probabilities):

        probabilities = np.asarray(probabilities, dtype=float)

        probabilities = np.clip(probabilities, 0.0, 1.0)

        return np.round(probabilities * 100).astype(int)

    def classify_risk(self, score):

        if score >= 85:
            return "CRITICAL"

        elif score >= 60:
            return "HIGH"

        elif score >= 30:
            return "MEDIUM"

        return "LOW"

    def get_action(self, tier):

        actions = {
            "CRITICAL": "Immediate investigation",
            "HIGH": "Priority analyst review",
            "MEDIUM": "Monitor transaction",
            "LOW": "Routine monitoring"
        }

        return actions.get(tier, "Manual review")

    def score_transactions(
        self,
        probabilities,
        transaction_data=None
    ):

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        scores = self.calculate_score(probabilities)

        tiers = [
            self.classify_risk(score)
            for score in scores
        ]

        flagged = (
            probabilities >= self.review_threshold
        ).astype(int)

        actions = [
            self.get_action(tier)
            for tier in tiers
        ]

        result = pd.DataFrame({
            "fraud_probability": probabilities,
            "risk_score": scores,
            "risk_tier": tiers,
            "review_flag": flagged,
            "recommended_action": actions
        })

        if transaction_data is not None:

            original = transaction_data.reset_index(
                drop=True
            ).copy()

            result = pd.concat(
                [original, result],
                axis=1
            )

        return result

    def summary(self, scored_data):

        total = len(scored_data)

        tier_counts = scored_data["risk_tier"].value_counts()

        summary = {
            "total_transactions": int(total),
            "flagged_transactions": int(
                scored_data["review_flag"].sum()
            ),
            "flagged_percentage": round(
                scored_data["review_flag"].mean() * 100,
                2
            ) if total else 0,
            "average_risk_score": round(
                scored_data["risk_score"].mean(),
                2
            ) if total else 0,
            "critical": int(
                tier_counts.get("CRITICAL", 0)
            ),
            "high": int(
                tier_counts.get("HIGH", 0)
            ),
            "medium": int(
                tier_counts.get("MEDIUM", 0)
            ),
            "low": int(
                tier_counts.get("LOW", 0)
            )
        }

        return summary
