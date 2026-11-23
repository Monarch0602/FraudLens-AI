
import numpy as np
import pandas as pd


class FraudFeatureEngineer:

    def __init__(self):
        self.is_fitted = False

    def fit(self, data):

        self.amount_mean = data["amount"].mean()
        self.amount_std = data["amount"].std()

        self.amount_95th_percentile = data["amount"].quantile(0.95)
        self.amount_sorted = np.sort(data["amount"].dropna().values)

        self.step_min = data["step"].min()
        self.step_max = data["step"].max()

        self.category_frequency = (
            data["category"].value_counts(normalize=True).to_dict()
        )

        self.category_amount_mean = (
            data.groupby("category")["amount"].mean().to_dict()
        )

        self.category_amount_std = (
            data.groupby("category")["amount"].std().fillna(0).to_dict()
        )

        self.is_fitted = True
        return self

    def transform(self, data):

        if not self.is_fitted:
            raise RuntimeError(
                "FeatureEngineer must be fitted before transformation."
            )

        df = data.copy()

        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["step"] = pd.to_numeric(df["step"], errors="coerce")

        # Amount features
        df["amount_log"] = np.log1p(df["amount"].clip(lower=0))

        if self.amount_std and self.amount_std > 0:
            df["amount_zscore"] = (
                df["amount"] - self.amount_mean
            ) / self.amount_std
        else:
            df["amount_zscore"] = 0.0

        # Percentile relative to fitted training distribution
        df["amount_percentile"] = (
            np.searchsorted(
                self.amount_sorted,
                df["amount"].fillna(self.amount_mean),
                side="right"
            ) / len(self.amount_sorted)
        )

        df["is_high_amount"] = (
            df["amount"] > self.amount_95th_percentile
        ).astype(int)

        # Step scaling
        step_range = self.step_max - self.step_min

        if step_range > 0:
            df["step_scaled"] = (
                df["step"] - self.step_min
            ) / step_range
        else:
            df["step_scaled"] = 0.0

        # Age code
        df["age_numeric"] = pd.to_numeric(
            df["age"], errors="coerce"
        )

        df["is_unknown_age"] = (
            df["age_numeric"].isna()
        ).astype(int)

        # Category-based features
        df["category_frequency"] = (
            df["category"]
            .map(self.category_frequency)
            .fillna(0)
        )

        df["category_amount_mean"] = (
            df["category"]
            .map(self.category_amount_mean)
            .fillna(self.amount_mean)
        )

        df["category_amount_std"] = (
            df["category"]
            .map(self.category_amount_std)
            .fillna(0)
        )

        # Gender encoding
        df["gender_encoded"] = (
            df["gender"]
            .astype(str)
            .str.upper()
            .map({"M": 1, "F": 0})
            .fillna(-1)
        )

        return df

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
