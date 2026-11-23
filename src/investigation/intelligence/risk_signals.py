
from typing import Any, Dict, List, Optional


class RiskSignalEngine:
    """
    Detects interpretable transaction risk indicators.

    Risk signals are investigation indicators, not proof of fraud.
    Statistical reference values should be calculated from training data
    or a trusted reference population.
    """

    def __init__(
        self,
        amount_95th_percentile: Optional[float] = None,
        amount_99th_percentile: Optional[float] = None,
        category_frequencies: Optional[Dict[str, float]] = None,
        rare_category_threshold: float = 0.01,
    ):
        self.amount_95th_percentile = amount_95th_percentile
        self.amount_99th_percentile = amount_99th_percentile

        self.category_frequencies = category_frequencies or {}
        self.rare_category_threshold = rare_category_threshold

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            result = float(value)

            if result != result:  # NaN check
                return None

            return result

        except (TypeError, ValueError):
            return None

    def analyze_transaction(
        self,
        transaction: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        signals = []

        amount = self._safe_float(transaction.get("amount"))
        category = transaction.get("category")
        age = transaction.get("age")
        gender = transaction.get("gender")

        # --------------------------------------------------
        # 1. Transaction amount signals
        # --------------------------------------------------

        if amount is None:
            signals.append({
                "signal": "invalid_amount",
                "description": "Transaction amount is missing or invalid.",
                "severity": "high",
                "points": 30,
            })

        elif amount < 0:
            signals.append({
                "signal": "negative_amount",
                "description": "Transaction amount is negative.",
                "severity": "high",
                "points": 30,
            })

        else:

            if (
                self.amount_99th_percentile is not None
                and amount >= self.amount_99th_percentile
            ):
                signals.append({
                    "signal": "extreme_amount",
                    "description": (
                        "Transaction amount is at or above "
                        "the reference 99th percentile."
                    ),
                    "severity": "high",
                    "points": 25,
                })

            elif (
                self.amount_95th_percentile is not None
                and amount >= self.amount_95th_percentile
            ):
                signals.append({
                    "signal": "high_amount",
                    "description": (
                        "Transaction amount is at or above "
                        "the reference 95th percentile."
                    ),
                    "severity": "medium",
                    "points": 15,
                })

        # --------------------------------------------------
        # 2. Transaction category signals
        # --------------------------------------------------

        if category is None or str(category).strip() == "":
            signals.append({
                "signal": "missing_category",
                "description": "Transaction category is missing.",
                "severity": "low",
                "points": 5,
            })

        else:

            category_key = str(category)

            if self.category_frequencies:

                if category_key not in self.category_frequencies:
                    signals.append({
                        "signal": "unknown_category",
                        "description": (
                            "Category was not present in "
                            "the reference data."
                        ),
                        "severity": "medium",
                        "points": 10,
                    })

                else:

                    frequency = self.category_frequencies[category_key]

                    if frequency < self.rare_category_threshold:
                        signals.append({
                            "signal": "rare_category",
                            "description": (
                                "Transaction category has a low "
                                "frequency in reference data."
                            ),
                            "severity": "low",
                            "points": 5,
                        })

        # --------------------------------------------------
        # 3. Age data quality signal
        # --------------------------------------------------

        if age is None or str(age).strip() == "":
            signals.append({
                "signal": "missing_age",
                "description": "Age information is missing.",
                "severity": "low",
                "points": 3,
            })

        # --------------------------------------------------
        # 4. Gender data quality signal
        # --------------------------------------------------

        if gender is None or str(gender).strip() == "":
            signals.append({
                "signal": "missing_gender",
                "description": "Gender information is missing.",
                "severity": "low",
                "points": 3,
            })

        return signals

    def analyze_batch(
        self,
        transactions: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:

        return [
            self.analyze_transaction(transaction)
            for transaction in transactions
        ]
