
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

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
N_SPLITS = 5

REPORT_DIR = Path("reports/model_evaluation")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. LOAD DATA
# =========================================================

print("=" * 70)
print("FRAUDLENS AI - 5-FOLD CROSS-VALIDATION")
print("=" * 70)

data = load_raw_data()

# =========================================================
# 2. KEEP THE TEST SET COMPLETELY SEPARATE
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

print("\nCross-validation will use the TRAINING partition only.")
print(f"Training samples available: {len(X_train):,}")
print(f"Fraud samples available: {int(y_train.sum()):,}")
print(f"Number of folds: {N_SPLITS}")

# =========================================================
# 3. FEATURE CONFIGURATION
# =========================================================

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

# =========================================================
# 4. CROSS-VALIDATION SETUP
# =========================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

trainer = FraudModelTrainer(random_state=RANDOM_STATE)

fold_results = []

# =========================================================
# 5. RUN CROSS-VALIDATION
# =========================================================

for fold_number, (train_indices, validation_indices) in enumerate(
    cv.split(X_train, y_train),
    start=1
):

    print("\n" + "=" * 70)
    print(f"FOLD {fold_number}/{N_SPLITS}")
    print("=" * 70)

    # -----------------------------------------------------
    # FOLD DATA
    # -----------------------------------------------------

    X_fold_train = X_train.iloc[train_indices].copy()
    X_fold_val = X_train.iloc[validation_indices].copy()

    y_fold_train = y_train.iloc[train_indices]
    y_fold_val = y_train.iloc[validation_indices]

    print(f"Fold training samples:   {len(X_fold_train):,}")
    print(f"Fold validation samples: {len(X_fold_val):,}")
    print(f"Training fraud:          {int(y_fold_train.sum())}")
    print(f"Validation fraud:        {int(y_fold_val.sum())}")

    # -----------------------------------------------------
    # FEATURE ENGINEERING - FIT ONLY ON FOLD TRAIN
    # -----------------------------------------------------

    feature_engineer = FraudFeatureEngineer()

    feature_engineer.fit(X_fold_train)

    X_fold_train_eng = feature_engineer.transform(
        X_fold_train
    )[selected_features]

    X_fold_val_eng = feature_engineer.transform(
        X_fold_val
    )[selected_features]

    # -----------------------------------------------------
    # CREATE FRESH MODELS FOR THIS FOLD
    # -----------------------------------------------------

    fold_models = trainer.build_models()

    # -----------------------------------------------------
    # TRAIN AND EVALUATE EACH MODEL
    # -----------------------------------------------------

    for model_name, model in fold_models.items():

        print(f"\nFold {fold_number} | Model: {model_name}")

        model.fit(
            X_fold_train_eng,
            y_fold_train
        )

        # Validation probabilities
        probabilities = model.predict_proba(
            X_fold_val_eng
        )[:, 1]

        # Optimize threshold using fold validation data
        optimizer = ThresholdOptimizer()

        threshold_results = optimizer.analyze(
            y_fold_val,
            probabilities
        )

        best_threshold_result = optimizer.best_threshold(
            threshold_results,
            metric="f1_score"
        )

        if isinstance(best_threshold_result, dict):
            threshold = float(best_threshold_result["threshold"])
        else:
            threshold = float(best_threshold_result.threshold)

        # Evaluate fold
        metrics = ModelEvaluator.evaluate(
            y_fold_val,
            probabilities,
            threshold=threshold
        )

        fold_results.append({
            "fold": fold_number,
            "model": model_name,
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1_score"],
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"]
            
        })

        print(f"Threshold: {threshold:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1:       {metrics['f1_score']:.4f}")
        print(f"PR-AUC:   {metrics['pr_auc']:.4f}")
        print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")

# =========================================================
# 6. FOLD-LEVEL RESULTS
# =========================================================

fold_df = pd.DataFrame(fold_results)

fold_path = REPORT_DIR / "cross_validation_folds.csv"

fold_df.to_csv(
    fold_path,
    index=False
)

# =========================================================
# 7. AGGREGATE RESULTS
# =========================================================

metrics_to_aggregate = [
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "threshold"
]

aggregate_rows = []

for model_name, group in fold_df.groupby("model"):

    row = {"model": model_name}

    for metric in metrics_to_aggregate:

        row[f"{metric}_mean"] = group[metric].mean()
        row[f"{metric}_std"] = group[metric].std(ddof=1)

    aggregate_rows.append(row)

aggregate_df = pd.DataFrame(aggregate_rows)

# Rank by mean PR-AUC
aggregate_df = aggregate_df.sort_values(
    by="pr_auc_mean",
    ascending=False
).reset_index(drop=True)

aggregate_path = REPORT_DIR / "cross_validation_summary.csv"

aggregate_df.to_csv(
    aggregate_path,
    index=False
)

# =========================================================
# 8. DISPLAY RESULTS
# =========================================================

print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

display_columns = [
    "model",
    "pr_auc_mean",
    "pr_auc_std",
    "f1_mean",
    "f1_std",
    "precision_mean",
    "recall_mean",
    "roc_auc_mean"
]

print(
    aggregate_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# =========================================================
# 9. BEST MODEL BASED ON CROSS-VALIDATION
# =========================================================

best_cv_model = aggregate_df.iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL FROM CROSS-VALIDATION")
print("=" * 70)

print(f"Model:       {best_cv_model['model']}")
print(f"Mean PR-AUC: {best_cv_model['pr_auc_mean']:.4f}")
print(f"PR-AUC Std:  {best_cv_model['pr_auc_std']:.4f}")
print(f"Mean F1:     {best_cv_model['f1_mean']:.4f}")
print(f"F1 Std:      {best_cv_model['f1_std']:.4f}")
print(f"Mean Recall:  {best_cv_model['recall_mean']:.4f}")
print(f"Mean ROC-AUC:{best_cv_model['roc_auc_mean']:.4f}")

# =========================================================
# 10. SAVE FINAL REPORT
# =========================================================

print("\nFiles generated:")

print(f"Fold results:    {fold_path}")
print(f"Summary results: {aggregate_path}")

print("\n5-fold cross-validation completed successfully.")
