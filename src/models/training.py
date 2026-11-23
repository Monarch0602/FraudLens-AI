
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.engineering import FraudFeatureEngineer


REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


class FraudModelTrainer:

    def __init__(self, random_state=42):

        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.feature_engineer = None

        self.numeric_features = [
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

        self.categorical_features = [
            "category"
        ]

    def prepare_features(self, X_train, X_test):

        # Learn feature statistics using training data only
        self.feature_engineer = FraudFeatureEngineer()
        self.feature_engineer.fit(X_train)

        train_engineered = self.feature_engineer.transform(X_train)
        test_engineered = self.feature_engineer.transform(X_test)

        # Exclude demographic features from the initial models
        train_engineered = train_engineered[
            self.numeric_features + self.categorical_features
        ]

        test_engineered = test_engineered[
            self.numeric_features + self.categorical_features
        ]

        return train_engineered, test_engineered

    def build_preprocessor(self):

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, self.numeric_features),
            ("categorical", categorical_pipeline, self.categorical_features)
        ])

        return preprocessor

    def build_models(self):

        preprocessor = self.build_preprocessor()

        models = {
            "Dummy Baseline": DummyClassifier(
                strategy="most_frequent"
            ),

            "Logistic Regression": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=self.random_state
                ))
            ]),

            "Random Forest": Pipeline([
                ("preprocessor", self.build_preprocessor()),
                ("classifier", RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ]),

            "Extra Trees": Pipeline([
                ("preprocessor", self.build_preprocessor()),
                ("classifier", ExtraTreesClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ])
        }

        return models

    def evaluate_model(self, name, model, X_test, y_test):

        predictions = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]
        else:
            probabilities = predictions.astype(float)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).ravel()

        results = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(y_test, probabilities),
            "PR_AUC": average_precision_score(y_test, probabilities),
            "False_Positive_Rate": (
                fp / (fp + tn) if (fp + tn) > 0 else 0
            ),
            "True_Positive": int(tp),
            "False_Positive": int(fp),
            "True_Negative": int(tn),
            "False_Negative": int(fn)
        }

        return results

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):

        print("\nPreparing features...")
        X_train_processed, X_test_processed = self.prepare_features(
            X_train, X_test
        )

        print("Building models...")
        models = self.build_models()

        print("\nTraining and evaluating models...")
        print("=" * 70)

        for name, model in models.items():

            print(f"\nTraining: {name}")

            # Dummy baseline needs no feature preprocessing
            if name == "Dummy Baseline":
                model.fit(X_train_processed, y_train)
                result = self.evaluate_model(
                    name, model, X_test_processed, y_test
                )

            else:
                model.fit(X_train_processed, y_train)
                result = self.evaluate_model(
                    name, model, X_test_processed, y_test
                )

            self.models[name] = model
            self.results[name] = result

            print(f"Precision: {result['Precision']:.4f}")
            print(f"Recall:    {result['Recall']:.4f}")
            print(f"F1:       {result['F1']:.4f}")
            print(f"PR-AUC:   {result['PR_AUC']:.4f}")

        self.save_results()

        return self.results

    def save_results(self):

        results_df = pd.DataFrame(self.results.values())

        results_df = results_df.sort_values(
            by="PR_AUC",
            ascending=False
        )

        output_path = REPORT_DIR / "model_comparison.csv"

        results_df.to_csv(output_path, index=False)

        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        print(results_df.to_string(index=False))

        print(f"\nResults saved to: {output_path}")
