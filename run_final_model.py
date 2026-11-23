
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

# ============================================================
# FRAUDLENS AI - FINAL MODEL TRAINING
# ============================================================

RANDOM_STATE = 42

MODEL_NAME = "Random Forest"

REPORT_DIR = Path("reports/model_evaluation")
MODEL_DIR = Path("models/saved")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = MODEL_DIR / "fraudlens_final_model.pkl"
FEATURE_ENGINEER_PATH = MODEL_DIR / "fraudlens_feature_engineer.pkl"
METADATA_PATH = MODEL_DIR / "fraudlens_model_metadata.json"

TEST_METRICS_PATH = REPORT_DIR / "final_test_metrics.json"
TEST_PREDICTIONS_PATH = REPORT_DIR / "final_test_predictions.csv"

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("FRAUDLENS AI - FINAL MODEL TRAINING")
print("=" * 70)

data = load_raw_data()

print(f"\nTotal transactions: {len(data):,}")
print(f"Total fraud: {int(data['fraud'].sum()):,}")
print(f"Fraud rate: {data['fraud'].mean() * 100:.2f}%")

# ============================================================
# 2. DATA SPLITTING
# ============================================================

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
print("DATA SPLIT")
print("=" * 70)

print(f"Training samples:   {len(X_train):,}")
print(f"Validation samples: {len(X_val):,}")
print(f"Test samples:      {len(X_test):,}")

print(f"Training fraud:    {int(y_train.sum()):,}")
print(f"Validation fraud:  {int(y_val.sum()):,}")
print(f"Test fraud:        {int(y_test.sum()):,}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

feature_engineer = FraudFeatureEngineer()

# Fit only on training data
feature_engineer.fit(X_train)

# Transform each partition
X_train_eng = feature_engineer.transform(X_train)
X_val_eng = feature_engineer.transform(X_val)
X_test_eng = feature_engineer.transform(X_test)

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

X_train_eng = X_train_eng[selected_features]
X_val_eng = X_val_eng[selected_features]
X_test_eng = X_test_eng[selected_features]

print(f"Selected features: {len(selected_features)}")

# ============================================================
# 4. BUILD FINAL RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("INITIALIZING FINAL MODEL")
print("=" * 70)

trainer = FraudModelTrainer(random_state=RANDOM_STATE)

models = trainer.build_models()

if MODEL_NAME not in models:
    raise KeyError(
        f"{MODEL_NAME} not found. Available models: "
        f"{list(models.keys())}"
    )

final_model = models[MODEL_NAME]

print(f"Selected model: {MODEL_NAME}")

# ============================================================
# 5. TRAIN ON TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)

final_model.fit(X_train_eng, y_train)

print("Training completed successfully.")

# ============================================================
# 6. VALIDATION - THRESHOLD SELECTION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 70)

val_probabilities = final_model.predict_proba(
    X_val_eng
)[:, 1]

threshold_optimizer = ThresholdOptimizer()

threshold_results = threshold_optimizer.analyze(
    y_val,
    val_probabilities
)

best_threshold_result = threshold_optimizer.best_threshold(
    threshold_results,
    metric="f1_score"
)

if isinstance(best_threshold_result, dict):
    best_threshold = float(
        best_threshold_result["threshold"]
    )
    validation_best_f1 = float(
        best_threshold_result["f1_score"]
    )
else:
    best_threshold = float(
        best_threshold_result["threshold"]
    )
    validation_best_f1 = float(
        best_threshold_result["f1_score"]
    )

validation_metrics = ModelEvaluator.evaluate(
    y_val,
    val_probabilities,
    threshold=best_threshold
)

print(f"Selected threshold: {best_threshold:.4f}")
print(f"Validation F1:    {validation_metrics['f1_score']:.4f}")
print(f"Validation PR-AUC:{validation_metrics['pr_auc']:.4f}")
print(f"Validation Recall:{validation_metrics['recall']:.4f}")
print(f"Validation Precision:{validation_metrics['precision']:.4f}")

# ============================================================
# 7. FINAL TEST EVALUATION
# ============================================================
# The model and threshold have already been selected.
# Test set is used only for final evaluation.

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_probabilities = final_model.predict_proba(
    X_test_eng
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_metrics = ModelEvaluator.evaluate(
    y_test,
    test_probabilities,
    threshold=best_threshold
)

# Confusion matrix counts
tp = int(((y_test == 1) & (test_predictions == 1)).sum())
tn = int(((y_test == 0) & (test_predictions == 0)).sum())
fp = int(((y_test == 0) & (test_predictions == 1)).sum())
fn = int(((y_test == 1) & (test_predictions == 0)).sum())

print("\nFINAL TEST RESULTS")
print("-" * 50)

print(f"Accuracy:           {test_metrics['accuracy']:.4f}")
print(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")
print(f"Precision:          {test_metrics['precision']:.4f}")
print(f"Recall:             {test_metrics['recall']:.4f}")
print(f"F1 Score:           {test_metrics['f1_score']:.4f}")
print(f"ROC-AUC:            {test_metrics['roc_auc']:.4f}")
print(f"PR-AUC:             {test_metrics['pr_auc']:.4f}")

print("\nConfusion Matrix Counts")
print("-" * 50)
print(f"True Positives:     {tp}")
print(f"True Negatives:     {tn}")
print(f"False Positives:    {fp}")
print(f"False Negatives:    {fn}")

print(f"\nFlagged transactions: {tp + fp}")

# ============================================================
# 8. SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL MODEL ARTIFACTS")
print("=" * 70)

# Save model pipeline
with open(FINAL_MODEL_PATH, "wb") as file:
    pickle.dump(final_model, file)

# Save feature engineering object
with open(FEATURE_ENGINEER_PATH, "wb") as file:
    pickle.dump(feature_engineer, file)

print(f"Model saved: {FINAL_MODEL_PATH}")
print(f"Feature engineer saved: {FEATURE_ENGINEER_PATH}")

# ============================================================
# 9. SAVE MODEL METADATA
# ============================================================

metadata = {
    "project": "FraudLens AI",
    "model_name": MODEL_NAME,
    "model_type": "Random Forest Classifier",
    "dataset": "BankSim",
    "random_state": RANDOM_STATE,

    "selection_method": "5-fold stratified cross-validation",
    "cross_validation_mean_pr_auc": 0.8250,
    "cross_validation_std_pr_auc": 0.0391,
    "cross_validation_mean_f1": 0.7559,
    "cross_validation_std_f1": 0.0411,

    "threshold_selection_method": "Validation F1 optimization",
    "decision_threshold": best_threshold,

    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "selected_features": selected_features,

    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),

    "validation_metrics": {
        key: float(value)
        for key, value in validation_metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    },

    "test_metrics": {
        key: float(value)
        for key, value in test_metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    },

    "test_confusion_counts": {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn
    },

    "test_set_used_for_model_selection": False
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4, default=str)

print(f"Metadata saved: {METADATA_PATH}")

# ============================================================
# 10. SAVE FINAL TEST METRICS
# ============================================================

final_metrics_report = {
    "model": MODEL_NAME,
    "threshold": best_threshold,
    "validation_f1": float(validation_metrics["f1_score"]),
    "validation_pr_auc": float(validation_metrics["pr_auc"]),
    "test_metrics": {
        key: float(value)
        for key, value in test_metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    },
    "confusion_counts": {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn
    }
}

with open(TEST_METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(final_metrics_report, file, indent=4)

print(f"Test metrics saved: {TEST_METRICS_PATH}")

# ============================================================
# 11. SAVE TEST PREDICTIONS
# ============================================================

test_predictions_df = X_test.copy()

test_predictions_df["actual_fraud"] = y_test.values
test_predictions_df["fraud_probability"] = test_probabilities
test_predictions_df["predicted_fraud"] = test_predictions
test_predictions_df["decision_threshold"] = best_threshold

test_predictions_df["risk_score"] = (
    test_probabilities * 100
).round(2)

test_predictions_df.to_csv(
    TEST_PREDICTIONS_PATH,
    index=False
)

print(f"Test predictions saved: {TEST_PREDICTIONS_PATH}")

# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FRAUDLENS AI - FINAL MODEL SUMMARY")
print("=" * 70)

print(f"Model:             {MODEL_NAME}")
print(f"Threshold:         {best_threshold:.4f}")
print(f"Validation F1:     {validation_metrics['f1_score']:.4f}")
print(f"Validation PR-AUC: {validation_metrics['pr_auc']:.4f}")
print(f"Test F1:          {test_metrics['f1_score']:.4f}")
print(f"Test PR-AUC:      {test_metrics['pr_auc']:.4f}")
print(f"Test Recall:       {test_metrics['recall']:.4f}")
print(f"Test Precision:    {test_metrics['precision']:.4f}")

print("\nGenerated files:")
print(f"- {FINAL_MODEL_PATH}")
print(f"- {FEATURE_ENGINEER_PATH}")
print(f"- {METADATA_PATH}")
print(f"- {TEST_METRICS_PATH}")
print(f"- {TEST_PREDICTIONS_PATH}")

print("\nFinal model pipeline completed successfully.")
