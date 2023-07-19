"""Preprocessing Methods"""
# system
from pathlib import Path

# math
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# package imports
import prototype.utils as u


# READ/LOAD DATASET
## read the dataset according dataContract and config definitions
def read_dataset(data_contract, config, date_type_key="data_type"):
    # Identify datatypes and excluded variables
    exclude_vars = u.get_list_vars(data_contract, "excluded_variables")
    str_vars = u.get_fields_by_keyvalue(
        data_contract,
        keyvalue=[date_type_key, "str"],
        exclude_vars=exclude_vars,
    )
    int_vars = u.get_fields_by_keyvalue(
        data_contract,
        keyvalue=[date_type_key, "int"],
        exclude_vars=exclude_vars,
    )
    float_vars = u.get_fields_by_keyvalue(
        data_contract,
        keyvalue=[date_type_key, "float"],
        exclude_vars=exclude_vars,
    )
    cat_vars = u.get_list_vars(data_contract, "categorical_variables")
    cat_vars += [value for value in str_vars if value not in cat_vars]
    num_vars = int_vars + float_vars
    date_vars = u.get_fields_by_keyvalue(
        data_contract,
        keyvalue=[date_type_key, "Date"],
        exclude_vars=exclude_vars,
    )
    date_formats = u.get_date_formats(data_contract, date_vars, exclude_vars)
    target_vars = u.get_list_vars(data_contract, "target_variables")

    # Identify dataset
    dataset = Path(config["dataset_location"] + config["dataset_name"])

    # Read csv file into a dataframe specifying separator and dtypes for each column
    df = pd.read_csv(
        dataset,
        sep=config["separator"],
        usecols=lambda x: x not in exclude_vars,
    )

    # Parse the date columns using pd.to_datetime() with format parameter
    # NOTE: doing after read_csv due to deprepactions from pandas v 2.0.0 date_vars and date_formats can be used in pd.read_csv
    for column, date_format in date_formats.items():
        df[column] = pd.to_datetime(df[column], format=date_format)

    return df, cat_vars, num_vars, target_vars




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
