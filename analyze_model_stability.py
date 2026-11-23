
from pathlib import Path

import pandas as pd
import numpy as np

# ============================================================
# FRAUDLENS AI - MODEL STABILITY ANALYSIS
# ============================================================

REPORT_DIR = Path("reports/model_evaluation")

FOLD_FILE = REPORT_DIR / "cross_validation_folds.csv"

print("=" * 70)
print("FRAUDLENS AI - MODEL STABILITY ANALYSIS")
print("=" * 70)

if not FOLD_FILE.exists():
    raise FileNotFoundError(
        f"Cross-validation file not found: {FOLD_FILE}"
    )

df = pd.read_csv(FOLD_FILE)

print("\nLoaded cross-validation results.")
print(f"Total records: {len(df)}")
print(f"Models: {df['model'].unique().tolist()}")
print(f"Folds: {sorted(df['fold'].unique())}")

# Metrics to analyze
metrics = [
    "pr_auc",
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "threshold",
]

# ============================================================
# 1. FOLD-BY-FOLD PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("FOLD-BY-FOLD PERFORMANCE")
print("=" * 70)

available_metrics = [m for m in metrics if m in df.columns]

fold_comparison = df[
    ["model", "fold"] + available_metrics
].sort_values(["model", "fold"])

print(fold_comparison.to_string(index=False))

fold_comparison.to_csv(
    REPORT_DIR / "stability_fold_comparison.csv",
    index=False
)

# ============================================================
# 2. STABILITY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("MODEL STABILITY STATISTICS")
print("=" * 70)

summary = df.groupby("model")[available_metrics].agg(
    ["mean", "std", "min", "max"]
)

print(summary.round(4).to_string())

summary.to_csv(
    REPORT_DIR / "model_stability_statistics.csv"
)

# ============================================================
# 3. COEFFICIENT OF VARIATION
# ============================================================

print("\n" + "=" * 70)
print("COEFFICIENT OF VARIATION (PR-AUC)")
print("=" * 70)

cv_results = df.groupby("model")["pr_auc"].agg(
    ["mean", "std"]
)

cv_results["coefficient_of_variation"] = (
    cv_results["std"] / cv_results["mean"]
)

cv_results["cv_percentage"] = (
    cv_results["coefficient_of_variation"] * 100
)

cv_results = cv_results.sort_values(
    "coefficient_of_variation"
)

print(cv_results.round(4).to_string())

cv_results.to_csv(
    REPORT_DIR / "model_pr_auc_stability.csv"
)

# ============================================================
# 4. MODEL PERFORMANCE RANGE
# ============================================================

print("\n" + "=" * 70)
print("PR-AUC PERFORMANCE RANGE")
print("=" * 70)

range_results = df.groupby("model")["pr_auc"].agg(
    minimum="min",
    maximum="max",
    mean="mean"
)

range_results["performance_range"] = (
    range_results["maximum"] - range_results["minimum"]
)

range_results = range_results.sort_values(
    "performance_range"
)

print(range_results.round(4).to_string())

range_results.to_csv(
    REPORT_DIR / "model_performance_range.csv"
)

# ============================================================
# 5. RANDOM FOREST VS EXTRA TREES
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST VS EXTRA TREES")
print("=" * 70)

rf = df[df["model"] == "Random Forest"]["pr_auc"].reset_index(drop=True)

et = df[df["model"] == "Extra Trees"]["pr_auc"].reset_index(drop=True)

if len(rf) == len(et):

    comparison = pd.DataFrame({
        "fold": range(1, len(rf) + 1),
        "random_forest_pr_auc": rf,
        "extra_trees_pr_auc": et,
    })

    comparison["difference_rf_minus_et"] = (
        comparison["random_forest_pr_auc"]
        - comparison["extra_trees_pr_auc"]
    )

    print(comparison.round(4).to_string(index=False))

    print("\nFold-wise winner:")
    print(
        comparison["difference_rf_minus_et"]
        .apply(
            lambda x: (
                "Random Forest"
                if x > 0
                else "Extra Trees"
                if x < 0
                else "Tie"
            )
        )
        .value_counts()
    )

    comparison.to_csv(
        REPORT_DIR / "rf_vs_extra_trees.csv",
        index=False
    )

# ============================================================
# 6. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("STABILITY ANALYSIS COMPLETED")
print("=" * 70)

print("\nGenerated reports:")

print("- stability_fold_comparison.csv")
print("- model_stability_statistics.csv")
print("- model_pr_auc_stability.csv")
print("- model_performance_range.csv")
print("- rf_vs_extra_trees.csv")

print("\nAll reports saved in:")
print(REPORT_DIR.resolve())
