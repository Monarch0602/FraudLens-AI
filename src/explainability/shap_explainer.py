
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class FraudExplainer:

    def __init__(self, model, feature_names=None):

        self.model = model
        self.feature_names = feature_names

        # Handle sklearn Pipeline objects
        if hasattr(model, "steps"):

            self.preprocessing = model[:-1]
            self.tree_model = model.steps[-1][1]

        else:

            self.preprocessing = None
            self.tree_model = model

        self.explainer = shap.TreeExplainer(self.tree_model)
        self.last_processed = None

    def _get_feature_names(self, n_features):

        # Use explicitly provided feature names if valid
        if (
            self.feature_names is not None
            and len(self.feature_names) == n_features
        ):
            return list(self.feature_names)

        # Retrieve names automatically from fitted sklearn preprocessing
        if self.preprocessing is not None:

            try:
                names = self.preprocessing.get_feature_names_out()

                if len(names) == n_features:

                    cleaned_names = []

                    for name in names:

                        name = str(name)

                        # Remove ColumnTransformer prefixes
                        name = name.replace("numeric__", "")
                        name = name.replace("categorical__", "")

                        # Make category names more readable
                        if name.startswith("category_es_"):
                            name = (
                                "Category: "
                                + name.replace("category_es_", "")
                            )

                        cleaned_names.append(name)

                    return cleaned_names

            except Exception:
                pass

        # Fallback if feature names cannot be recovered
        return [
            f"feature_{i}"
            for i in range(n_features)
        ]

    def _transform(self, X):

        if (
            self.preprocessing is not None
            and len(self.preprocessing.steps) > 0
        ):
            X_processed = self.preprocessing.transform(X)

        else:
            X_processed = X

        if isinstance(X_processed, pd.DataFrame):

            processed_df = X_processed.copy()

        else:

            if hasattr(X_processed, "toarray"):
                X_processed = X_processed.toarray()

            names = self._get_feature_names(
                X_processed.shape[1]
            )

            processed_df = pd.DataFrame(
                X_processed,
                columns=names
            )

        self.last_processed = processed_df

        return processed_df

    def _normalize_shap_values(self, shap_values):

        if isinstance(shap_values, list):

            if len(shap_values) > 1:
                return np.asarray(shap_values[1])

            return np.asarray(shap_values[0])

        values = np.asarray(shap_values)

        if values.ndim == 3:

            if values.shape[-1] > 1:
                return values[:, :, 1]

            return values[:, :, 0]

        return values

    def explain(self, X):

        X_processed = self._transform(X)

        raw_values = self.explainer.shap_values(
            X_processed
        )

        values = self._normalize_shap_values(
            raw_values
        )

        return values

    def global_importance(self, X, shap_values):

        X_processed = self.last_processed

        importance = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": X_processed.columns,
            "mean_abs_shap": importance
        })

        return importance_df.sort_values(
            "mean_abs_shap",
            ascending=False
        ).reset_index(drop=True)

    def plot_summary(self, X, shap_values, output_dir):

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        X_processed = self.last_processed

        plt.figure()

        shap.summary_plot(
            shap_values,
            X_processed,
            show=False,
            max_display=20
        )

        plt.tight_layout()

        plt.savefig(
            output_dir / "shap_summary.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

    def plot_bar(self, X, shap_values, output_dir):

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        X_processed = self.last_processed

        plt.figure()

        shap.summary_plot(
            shap_values,
            X_processed,
            plot_type="bar",
            show=False,
            max_display=20
        )

        plt.tight_layout()

        plt.savefig(
            output_dir / "shap_global_importance.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

    def explain_transaction(
        self,
        X,
        shap_values,
        index,
        output_dir
    ):

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        X_processed = self.last_processed

        row_values = shap_values[index]
        row_data = X_processed.iloc[index].values
        feature_names = list(X_processed.columns)

        explanation_df = pd.DataFrame({
            "feature": feature_names,
            "feature_value": row_data,
            "shap_value": row_values,
            "impact": np.where(
                row_values > 0,
                "Increases fraud risk",
                "Decreases fraud risk"
            )
        })

        explanation_df["absolute_impact"] = np.abs(
            explanation_df["shap_value"]
        )

        explanation_df = explanation_df.sort_values(
            "absolute_impact",
            ascending=False
        )

        explanation_df.to_csv(
            output_dir
            / f"transaction_{index}_explanation.csv",
            index=False
        )

        base_value = self.explainer.expected_value

        if isinstance(base_value, (list, np.ndarray)):

            base_value = np.asarray(
                base_value
            ).flatten()

            if len(base_value) > 1:
                base_value = base_value[1]
            else:
                base_value = base_value[0]

        explanation = shap.Explanation(
            values=row_values,
            base_values=float(base_value),
            data=row_data,
            feature_names=feature_names
        )

        plt.figure()

        shap.plots.waterfall(
            explanation,
            max_display=15,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / f"transaction_{index}_waterfall.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        return explanation_df
