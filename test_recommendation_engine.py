
from src.investigation.intelligence.recommendation_engine import (
    InvestigationRecommendationEngine,
)


engine = InvestigationRecommendationEngine()


test_cases = [
    {
        "name": "Normal transaction",
        "fraud_probability": 0.05,
        "signals": [],
    },
    {
        "name": "High amount",
        "fraud_probability": 0.45,
        "signals": [
            {
                "signal": "high_amount",
                "severity": "medium",
                "points": 15,
            }
        ],
    },
    {
        "name": "Extreme amount + rare category",
        "fraud_probability": 0.92,
        "signals": [
            {
                "signal": "extreme_amount",
                "severity": "high",
                "points": 25,
            },
            {
                "signal": "rare_category",
                "severity": "low",
                "points": 5,
            },
        ],
    },
    {
        "name": "Unknown category + missing data",
        "fraud_probability": 0.75,
        "signals": [
            {
                "signal": "unknown_category",
                "severity": "medium",
                "points": 10,
            },
            {
                "signal": "missing_age",
                "severity": "low",
                "points": 3,
            },
            {
                "signal": "missing_gender",
                "severity": "low",
                "points": 3,
            },
        ],
    },
]


for test_case in test_cases:

    print("\n" + "=" * 65)
    print(f"TEST: {test_case['name']}")
    print("=" * 65)

    recommendations = engine.generate_recommendations(
        signals=test_case["signals"],
        fraud_probability=test_case["fraud_probability"],
    )

    if not recommendations:
        print("No investigation recommendations generated.")

    for recommendation in recommendations:

        print(
            f"[{recommendation['priority'].upper()}] "
            f"{recommendation['action']}"
        )

        print(
            f"Reason: {recommendation['reason']}"
        )

        print(
            f"Source: {recommendation['source_signal']}"
        )

        print("-" * 50)
