"""Preprocessing Methods"""
# system
from pathlib import Path

# math
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


"""
data Preprocessing
"""


# TRANSFORMATIONS


## Clean wrong characters in numerical columns and teransform them to numeric
class ConvertDataTypes(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.numerical_cols:
            X[col].replace(regex=True, inplace=True, to_replace=r"[^0-9.\-]", value=r"")

        # Convert columns to numeric that were read as object.
        X[self.numerical_cols] = X[self.numerical_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        X[self.categorical_cols] = X[self.categorical_cols].astype("category")

        return X


## Drop duplicates
class DropDuplicates(BaseEstimator, TransformerMixin):
    def __init__(self, keep="first"):
        self.keep = keep

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop_duplicates(keep=self.keep)


## Impute missing values (mean, median, most frequent)
class DataImputing(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="mean", impute_cols=None):
        self.strategy = strategy
        self.impute_cols = impute_cols

    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(X[self.impute_cols])
        return self

    def transform(self, X):
        X[self.impute_cols] = self.imputer.transform(X[self.impute_cols])
        return X
