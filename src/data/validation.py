
from pathlib import Path
import pandas as pd
import numpy as np


class DataValidator:

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.target_column = self._detect_target()

    def _detect_target(self):
        possible_targets = [
            "fraud",
            "isFraud",
            "is_fraud",
            "fraud_label",
            "label",
            "target"
        ]

        for column in self.data.columns:
            if column.lower() in [t.lower() for t in possible_targets]:
                return column

        return None

    def validate(self):

        report = {}

        # Dataset overview
        report["dataset_overview"] = {
            "rows": int(self.data.shape[0]),
            "columns": int(self.data.shape[1]),
            "memory_usage_mb": round(
                self.data.memory_usage(deep=True).sum() / (1024 ** 2), 2
            )
        }

        # Missing values
        missing_values = self.data.isnull().sum()

        report["missing_values"] = {
            column: int(count)
            for column, count in missing_values.items()
            if count > 0
        }

        report["total_missing_values"] = int(missing_values.sum())

        # Duplicate rows
        report["duplicate_rows"] = int(self.data.duplicated().sum())

        # Data types
        report["data_types"] = {
            column: str(dtype)
            for column, dtype in self.data.dtypes.items()
        }

        # Numeric features
        numeric_columns = self.data.select_dtypes(
            include=np.number
        ).columns.tolist()

        report["numeric_features"] = numeric_columns

        # Categorical features
        categorical_columns = self.data.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        report["categorical_features"] = categorical_columns

        # Numeric statistics
        if numeric_columns:
            report["numeric_statistics"] = (
                self.data[numeric_columns]
                .describe()
                .replace({np.nan: None})
                .to_dict()
            )

        # Target analysis
        if self.target_column:

            target = self.data[self.target_column]

            report["target_column"] = self.target_column

            report["target_distribution"] = {
                str(key): int(value)
                for key, value in target.value_counts(dropna=False).items()
            }

            report["target_percentage"] = {
                str(key): round(float(value / len(target) * 100), 4)
                for key, value in target.value_counts(dropna=False).items()
            }

            if target.nunique() == 2:

                counts = target.value_counts()

                minority_class = counts.min()
                majority_class = counts.max()

                report["class_imbalance_ratio"] = round(
                    float(majority_class / minority_class), 2
                ) if minority_class > 0 else None

                report["class_imbalance_detected"] = bool(
                    majority_class / minority_class >= 5
                ) if minority_class > 0 else False

        else:
            report["target_column"] = None
            report["target_warning"] = (
                "Fraud target column was not automatically identified."
            )

        # Potential constant columns
        constant_columns = [
            column
            for column in self.data.columns
            if self.data[column].nunique(dropna=False) <= 1
        ]

        report["constant_columns"] = constant_columns

        # Potential high-cardinality features
        high_cardinality_columns = [
            column
            for column in categorical_columns
            if self.data[column].nunique() > 100
        ]

        report["high_cardinality_columns"] = high_cardinality_columns

        # Amount-specific analysis
        amount_columns = [
            column for column in self.data.columns
            if "amount" in column.lower()
        ]

        report["amount_analysis"] = {}

        for column in amount_columns:

            if pd.api.types.is_numeric_dtype(self.data[column]):

                report["amount_analysis"][column] = {
                    "min": float(self.data[column].min()),
                    "max": float(self.data[column].max()),
                    "mean": float(self.data[column].mean()),
                    "median": float(self.data[column].median()),
                    "negative_values": int(
                        (self.data[column] < 0).sum()
                    ),
                    "zero_values": int(
                        (self.data[column] == 0).sum()
                    )
                }

        # Data quality warnings
        warnings = []

        if report["total_missing_values"] > 0:
            warnings.append("Dataset contains missing values.")

        if report["duplicate_rows"] > 0:
            warnings.append("Dataset contains duplicate rows.")

        if report["constant_columns"]:
            warnings.append("Dataset contains constant columns.")

        if report["target_column"] is None:
            warnings.append("Fraud target column could not be detected.")

        if report.get("class_imbalance_detected", False):
            warnings.append("Severe class imbalance detected.")

        report["warnings"] = warnings

        report["validation_status"] = (
            "WARNING" if warnings else "PASSED"
        )

        return report

    def save_report(self, report, output_path="reports/data_validation_report.json"):

        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, default=str)

        print(f"Validation report saved to: {output_path}")
