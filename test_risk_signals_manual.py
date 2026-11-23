
from src.investigation.intelligence.risk_signals import RiskSignalEngine


# Example reference statistics
engine = RiskSignalEngine(
    amount_95th_percentile=8000,
    amount_99th_percentile=20000,
    category_frequencies={
        "food": 0.35,
        "grocery": 0.22,
        "transport": 0.15,
        "electronics": 0.005,
    },
    rare_category_threshold=0.01,
)


transactions = [
    {
        "amount": 500,
        "category": "food",
        "age": "2",
        "gender": "M",
    },
    {
        "amount": 9000,
        "category": "food",
        "age": "3",
        "gender": "F",
    },
    {
        "amount": 25000,
        "category": "electronics",
        "age": "4",
        "gender": "M",
    },
    {
        "amount": 500,
        "category": "unknown_category",
        "age": None,
        "gender": None,
    },
]


for index, transaction in enumerate(transactions, start=1):

    signals = engine.analyze_transaction(transaction)

    print("=" * 60)
    print(f"TRANSACTION {index}")
    print("=" * 60)

    if not signals:
        print("No risk signals detected.")

    else:

        for signal in signals:
            print(f"Signal:    {signal['signal']}")
            print(f"Severity:  {signal['severity']}")
            print(f"Points:    {signal['points']}")
            print(f"Reason:    {signal['description']}")
            print()
