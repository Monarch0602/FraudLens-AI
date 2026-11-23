
import inspect

from src.explainability.shap_explainer import FraudExplainer

print("=" * 60)
print("FRAUDLENS AI - MODEL PIPELINE INSPECTION")
print("=" * 60)

print("\nFraudExplainer location:")
print(inspect.getfile(FraudExplainer))

print("\nFraudExplainer methods:")
for name, method in inspect.getmembers(FraudExplainer, predicate=inspect.isfunction):
    print(f"\n--- {name} ---")
    try:
        print(inspect.signature(method))
    except Exception:
        pass

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)
