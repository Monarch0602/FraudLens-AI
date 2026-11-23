
from pathlib import Path
import json
import pickle
import warnings

import pandas as pd
import numpy as np

from src.data.loader import load_raw_data
from src.models.splitting import split_three_way
from src.models.training import FraudModelTrainer
from src.features.engineering import FraudFeatureEngineer

from src.evaluation.model_evaluator import ModelEvaluator
from src.evaluation.threshold_optimizer import ThresholdOptimizer

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42

REPORT_DIR = Path("reports/model_evaluation")
MODEL_DIR = Path("models/saved")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. LOAD DATA
# =========================================================

print("\n" + "=" * 70)
print("FRAUDLENS AI - VALIDATION-BASED MODEL EVALUATION")
print("=" * 70)

data = load_raw_data()

print("\nDataset information:")
print(f"Total transactions: {len(data):,}")
print(f"Fraud transactions: {int(data['fraud'].sum()):,}")
print(f"Legitimate transactions: {int((data['fraud'] == 0).sum()):,}")
print(f"Fraud rate: {data['fraud'].mean() * 100:.4f}%")

# =========================================================
# 2. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# =========================================================

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
) = split_three_way(
    data,
    target="fraud",
    test_size=0.20,
    validation_size=0.20,
    random_state=RANDOM_STATE
)

print("\n" + "=" * 70)
print("THREE-WAY DATA SPLIT")
print("=" * 70)

for name, target in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test)
]:
    print(f"\n{name}: {len(target):,} samples")
    print(f"Fraud distribution: {target.value_counts().to_dict()}")
    print(f"Fraud rate: {target.mean() * 100:.2f}%")

# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

feature_engineer = FraudFeatureEngineer()

# Fit feature engineering ONLY on training data
feature_engineer.fit(X_train)

# Apply the fitted transformations to all sets
X_train_engineered = feature_engineer.transform(X_train)
X_val_engineered = feature_engineer.transform(X_val)
X_test_engineered = feature_engineer.transform(X_test)

numeric_features = [
    "amount",
    "amount_log",
    "amount_zscore",
    "amount_percentile",
    "is_high_amount",
    "step_scaled",
    "category_frequency",
    "category_amount_mean",
    "category_amount_std"
]

categorical_features = ["category"]

selected_features = numeric_features + categorical_features

X_train_engineered = X_train_engineered[selected_features]
X_val_engineered = X_val_engineered[selected_features]
X_test_engineered = X_test_engineered[selected_features]

print(f"Engineered features: {len(selected_features)}")
print(f"Training samples: {len(X_train_engineered):,}")
print(f"Validation samples: {len(X_val_engineered):,}")
print(f"Testing samples: {len(X_test_engineered):,}")

# =========================================================
# 4. BUILD MODELS
# =========================================================

print("\n" + "=" * 70)
print("MODEL INITIALIZATION")
print("=" * 70)

trainer = FraudModelTrainer(random_state=RANDOM_STATE)
models = trainer.build_models()

print(f"\nModels selected: {len(models)}")

# =========================================================
# 5. TRAIN MODELS AND SELECT USING VALIDATION
# =========================================================

comparison_results = []
threshold_results = []
trained_models = {}

for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Training: {model_name}")
    print("-" * 70)

    # Models already contain their own preprocessing
    model.fit(X_train_engineered, y_train)

    # -----------------------------------------------------
    # VALIDATION PREDICTIONS
    # -----------------------------------------------------

    val_probabilities = model.predict_proba(
        X_val_engineered
    )[:, 1]

    # -----------------------------------------------------
    # THRESHOLD OPTIMIZATION ON VALIDATION SET ONLY
    # -----------------------------------------------------

    threshold_analyzer = ThresholdOptimizer()

    threshold_grid = threshold_analyzer.analyze(
        y_val,
        val_probabilities
    )

    best_threshold_result = threshold_analyzer.best_threshold(
        threshold_grid,
        metric="f1_score"
    )

    if isinstance(best_threshold_result, dict):
        best_threshold = float(best_threshold_result["threshold"])
        validation_best_f1 = float(best_threshold_result["f1_score"])
    else:
        best_threshold = float(best_threshold_result["threshold"])
        validation_best_f1 = float(best_threshold_result["f1_score"])

    # Validation metrics
    validation_metrics = ModelEvaluator.evaluate(
        y_val,
        val_probabilities,
        threshold=best_threshold
    )

    validation_pr_auc = float(validation_metrics["pr_auc"])
    validation_roc_auc = float(validation_metrics["roc_auc"])

    print(f"Validation threshold: {best_threshold:.4f}")
    print(f"Validation F1:      {validation_best_f1:.4f}")
    print(f"Validation PR-AUC:  {validation_pr_auc:.4f}")
    print(f"Validation ROC-AUC: {validation_roc_auc:.4f}")

    # -----------------------------------------------------
    # TEST EVALUATION
    # -----------------------------------------------------
    # Test data is used for reporting only.
    # It is NOT used to select the best model.

    test_probabilities = model.predict_proba(
        X_test_engineered
    )[:, 1]

    test_metrics = ModelEvaluator.evaluate(
        y_test,
        test_probabilities,
        threshold=best_threshold
    )

    # Calculate confusion matrix counts directly
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    tp = int(((y_test == 1) & (test_predictions == 1)).sum())
    tn = int(((y_test == 0) & (test_predictions == 0)).sum())
    fp = int(((y_test == 0) & (test_predictions == 1)).sum())
    fn = int(((y_test == 1) & (test_predictions == 0)).sum())

    # -----------------------------------------------------
    # SAVE COMPARISON RESULTS
    # -----------------------------------------------------

    comparison_results.append({
        "model": model_name,
        "threshold": best_threshold,

        "validation_precision": validation_metrics["precision"],
        "validation_recall": validation_metrics["recall"],
        "validation_f1": validation_metrics["f1_score"],
        "validation_pr_auc": validation_pr_auc,
        "validation_roc_auc": validation_roc_auc,

        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1_score"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],

        "test_true_positives": tp,
        "test_true_negatives": tn,
        "test_false_positives": fp,
        "test_false_negatives": fn,
        "test_flagged_transactions": tp + fp
    })

    # -----------------------------------------------------
    # SAVE THRESHOLD ANALYSIS
    # -----------------------------------------------------

    if isinstance(threshold_grid, pd.DataFrame):
        threshold_df = threshold_grid.copy()
    else:
        threshold_df = pd.DataFrame(threshold_grid)

    threshold_df["model"] = model_name
    threshold_results.append(threshold_df)

    # -----------------------------------------------------
    # SAVE TRAINED MODEL
    # -----------------------------------------------------

    model_filename = (
        str(model_name)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    model_path = MODEL_DIR / f"{model_filename}.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    trained_models[model_name] = {
        "model": model,
        "threshold": best_threshold,
        "validation_pr_auc": validation_pr_auc,
        "validation_f1": validation_best_f1,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics
    }

    print("\nTest results:")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Precision:   {test_metrics['precision']:.4f}")
    print(f"Recall:      {test_metrics['recall']:.4f}")
    print(f"F1 Score:   {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:    {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:     {test_metrics['pr_auc']:.4f}")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

# =========================================================
# 6. MODEL COMPARISON
# =========================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame(comparison_results)

# IMPORTANT:
# Select the best model using VALIDATION PR-AUC.
# Test metrics must never be used for model selection.

comparison_df = comparison_df.sort_values(
    by="validation_pr_auc",
    ascending=False
).reset_index(drop=True)

comparison_path = REPORT_DIR / "model_comparison.csv"

comparison_df.to_csv(
    comparison_path,
    index=False
)

display_columns = [
    "model",
    "threshold",
    "validation_f1",
    "validation_pr_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_pr_auc",
    "test_roc_auc",
    "test_false_positives",
    "test_false_negatives"
]

print(
    comparison_df[display_columns].to_string(index=False)
)

# =========================================================
# 7. THRESHOLD ANALYSIS EXPORT
# =========================================================

if threshold_results:

    all_thresholds = pd.concat(
        threshold_results,
        ignore_index=True
    )

    threshold_path = REPORT_DIR / "threshold_analysis.csv"

    all_thresholds.to_csv(
        threshold_path,
        index=False
    )

    print(f"\nThreshold analysis saved: {threshold_path}")

# =========================================================
# 8. SAVE EVALUATION SUMMARY
# =========================================================

summary = {
    "project": "FraudLens AI",
    "dataset": "BankSim",
    "random_state": RANDOM_STATE,
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "fraud_rate": float(data["fraud"].mean()),
    "selection_metric": "validation_pr_auc",
    "test_set_used_for_model_selection": False,
    "models": comparison_results
}

json_path = REPORT_DIR / "evaluation_summary.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        indent=4,
        default=str
    )

# =========================================================
# 9. BEST MODEL - SELECTED USING VALIDATION PR-AUC
# =========================================================

best_model_row = comparison_df.iloc[0]

best_model_info = {
    "model": str(best_model_row["model"]),
    "selection_metric": "validation_pr_auc",
    "validation_pr_auc": float(best_model_row["validation_pr_auc"]),
    "validation_f1": float(best_model_row["validation_f1"]),
    "threshold": float(best_model_row["threshold"]),

    # Test metrics are reported only after model selection
    "test_pr_auc": float(best_model_row["test_pr_auc"]),
    "test_f1": float(best_model_row["test_f1"]),
    "test_precision": float(best_model_row["test_precision"]),
    "test_recall": float(best_model_row["test_recall"]),
    "test_roc_auc": float(best_model_row["test_roc_auc"]),

    "test_false_positives": int(
        best_model_row["test_false_positives"]
    ),
    "test_false_negatives": int(
        best_model_row["test_false_negatives"]
    )
}

best_model_path = REPORT_DIR / "best_model_summary.json"

with open(best_model_path, "w", encoding="utf-8") as file:
    json.dump(
        best_model_info,
        file,
        indent=4
    )

# =========================================================
# 10. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nBest model selected using VALIDATION PR-AUC:")
print(f"Model:           {best_model_info['model']}")
print(f"Validation PR-AUC: {best_model_info['validation_pr_auc']:.4f}")
print(f"Validation F1:    {best_model_info['validation_f1']:.4f}")
print(f"Test PR-AUC:      {best_model_info['test_pr_auc']:.4f}")
print(f"Test F1:          {best_model_info['test_f1']:.4f}")
print(f"Test Precision:   {best_model_info['test_precision']:.4f}")
print(f"Test Recall:      {best_model_info['test_recall']:.4f}")
print(f"Threshold:        {best_model_info['threshold']:.4f}")

print("\nFiles generated:")
print(f"Comparison CSV: {comparison_path}")
print(f"Threshold CSV:  {REPORT_DIR / 'threshold_analysis.csv'}")
print(f"Summary JSON:   {json_path}")
print(f"Best model:     {best_model_path}")
print(f"Saved models:   {MODEL_DIR}")

print("\nFraudLens AI evaluation pipeline finished.")
