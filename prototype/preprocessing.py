"""Preprocessing Methods"""
# system
from pathlib import Path
import platform
import sys

# math
import pandas as pd
import numpy as np
import scipy
import scipy.stats as ss
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from dataprep.eda import create_report
from datetime import datetime
import os
import warnings
import gc
from sklearn.impute import KNNImputer
from matplotlib.patches import Rectangle
from pandas.tseries.frequencies import to_offset


# nn training
import torch

# package imports
import prototype.utils as u


"""
Data Ingestion
"""

# CONFIGURATIONS


# Prepare the inference dataset according to the configuration file, applying resampling and rolling window
def config_infer(df, config, target_vars):
    # Apply all the functions defined in config_prep related to inference sequentially

    # Group by columns and calculate the mean of the aggregated columns
    # if 'groupby_columns' in config and config['groupby_columns'] is not None:
    #     df = df.groupby(config['groupby_columns']).agg({col: config['aggregation_method'] for col in config['aggregation_columns']})

    # Resample the data to the frequency specified in the configuration
    if "frequency" in config and config["frequency"] is not None:
        df = df[target_vars].resample(config["frequency"]).mean()

    # Apply the rolling window to the data
    if "window_method" in config and config["window_method"] is not None:
        for target_var in target_vars:
            df["Rolling_" + target_var] = (
                df[target_var].rolling(config["inference_minimum_window_size"]).mean()
            )

    return df


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


def read_driver_based(data_contract):
    exclude_vars = u.get_list_vars(data_contract, "excluded_variables")
    driven = u.get_fields_by_keyvalue(
        data_contract,
        keyvalue=["driver_based_type", "driven"],
        exclude_vars=exclude_vars,
    )
    drivers = u.get_fields_by_keyvalue(
        data_contract,
        keyvalue=["driver_based_type", "driver"],
        exclude_vars=exclude_vars,
    )

    return driven, drivers


"""
data Summary
"""


def summary(df):
    summ = pd.DataFrame(
        {
            "Variables": df.columns.tolist(),
            "Missing Values": df.isnull().sum(),
            "Unique Values": df.nunique(),
        }
    ).reset_index(drop=True)

    return summ


def list_category_combinations(df, cols_cat):
    # List the number of classes and combinations from categorical variables
    list_classes = []
    comb = 1
    for c in cols_cat:
        classes = df[c].unique()
        n_class = df[c].nunique()  # + (1) #agregation ? ??
        list_classes.append(n_class)
        comb = comb * n_class
        codes, uniques = pd.factorize(df[c])
        print(
            f"{c}: {n_class} categories; factorization {uniques}"
        )  # f"{c}: {n_class} categories;  first: {classes[0]}, last: {classes[-1]};"
    print(f"Combinations: {comb}")


def list_unique_combinations(df, cols_cat):
    df = df[cols_cat].copy()
    df = df.groupby(cols_cat, as_index=False).size()
    df = df.sort_values(by="size").reset_index(drop=True)

    return df


def summ_numerical(df, cols_num):
    df_num = df.loc[:, cols_num]
    summ_num = pd.DataFrame(index=cols_num)
    summ_num["Data Type"] = df_num.dtypes.values
    summ_num["# Non-null Records"] = df_num.count().values
    summ_num["# Non-Zero Records"] = df_num.astype(bool).sum(axis=0)
    summ_num["% Populated"] = round(
        summ_num["# Non-null Records"] / df_num.shape[0] * 100, 2
    )
    summ_num["# Unique Values"] = df_num.nunique().values
    summ_num["Mean"] = round(df_num.mean(numeric_only=True), 2)
    summ_num["Std"] = round(df_num.std(numeric_only=True), 2)
    summ_num["Min"] = round(df_num.min(numeric_only=True), 2)
    summ_num["Max"] = round(df_num.max(numeric_only=True), 2)
    # describe = df.describe()

    return summ_num


def summ_categorical(df, cols_cat):
    df_cat = df[cols_cat]
    summ_cat = pd.DataFrame(index=df_cat.columns)
    summ_cat["Data Type"] = df_cat.dtypes.values
    summ_cat["# Non-null Records"] = df_cat.count().values
    summ_cat["% Populated"] = round(
        summ_cat["# Non-null Records"] / df_cat.shape[0] * 100, 2
    )
    summ_cat["# Unique Values"] = df_cat.nunique().values

    temp = []
    for col in df_cat.columns:
        temp.append(df_cat[col].value_counts().idxmax())
    summ_cat["Most Common Values"] = temp

    summ_unique = list_unique_combinations(df, cols_cat)
    summ_combi = pd.DataFrame(
        {
            "# Possible combinations": [summ_cat["# Unique Values"].prod()],
            "# Unique combinations": [summ_unique.shape[0]],
        }
    )
    summ_combi = summ_combi.T

    return summ_cat, summ_combi, summ_unique


# Analyze the data using dataprep.eda
def analyze_dataset(df, cols_cat, report_title, report_path, save=False):
    # suppress warnings
    warnings.filterwarnings("ignore")

    # Create a copy of the dataframe to analyze and convert the categorical columns to category just to visualization purposes
    print(df.shape)
    df_analyze = df.copy()
    df_analyze[cols_cat] = df_analyze[cols_cat].astype("category")

    report = create_report(df_analyze, title=report_title)

    report.show_browser()

    # Check if the directory exists to save the report
    if not os.path.exists(report_path):
        # If the directory does not exist, create it
        os.makedirs(report_path)
    filename = report_path + report_title + ".html"
    if save:
        report.save(filename)

    # Force delete the dataframe to free memory
    del df_analyze
    gc.collect()

    return


"""
data Preprocessing
"""


# TRANSFORMATIONS


## Drop the columns in dataframe passed as a parameter
## TranformerMixin implement automatically fit_transform method without the need to implement it manually
## This classes are used for a global dataset in an usupervised learning. If we split the dataset in train and test we  must fit on the training and fittransform on the train and test datasets
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input is not a pandas DataFrame")

        # Drop columns
        X = X.drop(columns=self.columns)

        return X


## Clean wrong characters in numerical columns and teransform them to numeric
class TransformNumericals(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols):
        self.numerical_cols = numerical_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.numerical_cols:
            X[col].replace(regex=True, inplace=True, to_replace=r"[^0-9.\-]", value=r"")

        # Convert columns to numeric that were read as object.
        X[self.numerical_cols] = X[self.numerical_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        return X


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


## Delete columns and rows with missing values above a certain threshold
class DropNaN(BaseEstimator, TransformerMixin):
    def __init__(self, col_threshold=0.3, row_threshold=0.3):
        self.col_threshold = col_threshold
        self.row_threshold = row_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop columns with missing values above threshold (this threshold identify the number of non NaN values)
        X = X.dropna(axis=1, thresh=int(X.shape[0] * (1 - self.col_threshold)))

        # Drop rows with missing values above threshold
        row_na_count = X.isna().sum(axis=1)
        row_na_percent = row_na_count / X.shape[1]
        X = X[row_na_percent < self.row_threshold]
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


## Encode categorical variables using LabelEncoder or OneHotEncoder
class DataEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, method="label"):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode categorical variables using specified method
        categorical_cols = X.select_dtypes(
            exclude=[np.number, "datetime64[ns]"]
        ).columns

        if self.method == "label":
            label_encoder = LabelEncoder()
            for col in categorical_cols:
                X[col] = label_encoder.fit_transform(X[col])
            return X
        else:
            # categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
            encoded_data = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            return encoded_data


## Use Standard scaler or MinMaxScaler to scale numerical variables
class DataScaling(BaseEstimator, TransformerMixin):
    def __init__(self, method="standard", target_columns=None):
        self.method = method
        self.target_columns = target_columns

    def fit(self, X, y=None):
        if self.method == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        self.scaler.fit(X[self.target_columns])
        return self

    def transform(self, X):
        X[self.target_columns] = self.scaler.transform(X[self.target_columns])
        return X


## Analysis of missing values


def display_nan_stats(df):
    # Function to display the NaN statistics
    # Calculate the total number of NaN values in each column
    total = df.isnull().sum()

    # Calculate the percentage of NaN values in each column
    percent = (df.isnull().sum() / df.isnull().count()) * 100

    # Create a DataFrame to display the results
    result = pd.concat(
        [total, percent], axis=1, keys=["Total of NaN", "Percent of NaN"]
    )
    print(result)


def check_nans(df, nan_c):
    for c in nan_c:
        nans_nr = df[c].isna().sum()
        print(f"{c} : {nans_nr}, {100*nans_nr/df.shape[0]:.2f}%")


def get_nans_columns(df, check=False):
    # Identify missing values: any nans present?
    nan_c = df.columns[df.isna().any()].tolist()
    if check:
        check_nans(df, nan_c)
    return nan_c


def nans_transformer(df, nan_v=None, check=False):
    # Identify missing values: any nans present?
    nan_c = get_nans_columns(df, check=check)
    # Analysis of nans:  ?with -1, drop nans row?, or do imputation of NANs
    if nan_v is not None:
        previous_shape = df.shape[0]
        df = df.fillna(nan_v)
        # TODO: imputation of NANs
    else:
        nan_v = np.nan
        df = df.dropna()

    # Check nans again
    if check:
        print("After nan transformer:")
        check_nans(df, nan_c)
        print(f"dfset: {100*df.shape[0]/previous_shape:.2f}%")
        # check distribution of nans
        print("NAN distributions:")
        for c in nan_c:
            print(c, np.sum(df[c] == nan_v))
            df[df[c] == nan_v].hist(bins=50, figsize=(15, 10))

    return df


def dtype_transformer(df, cols_num, cols_cat):
    # df Integrety & Conversion: change dtype where necessary

    # Unicode str object to numeric
    for col in cols_num:
        # WARNING: use always .loc or .iloc to  avoid chained indexing: https://pandas.pydf.org/docs/user_guide/indexing.html#indexing-view-versus-copy
        df.loc[:, col] = (
            df[col].astype(str).str.replace(" ", "")
        )  # removing trailing spaces in numeric str
        df.loc[:, col] = pd.to_numeric(
            df[col], errors="raise", downcast="integer"
        )  # needs to raise error because of trailing spaces
    # Categorical to ordinal
    for col in cols_cat:
        df.loc[:, col] = pd.factorize(df[col])[0]  # this already uses NaN sentinel

    return df


# Calculate the missing periods in a time series
def sparsity_ts(ts, freq):
    # Create a date range that covers the entire time period of interest
    date_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq=freq)

    # Calculate the number of missing periods
    num_missing_periods = len(date_range) - len(ts)

    return num_missing_periods


## Aggregators (sum, mean,...)


def df_sum_aggregator(df, by=["Start Date", "Start Hour"]):
    """Sum numerical columns"""
    df = df.groupby(by=by).sum().reset_index()
    return df


# Fill missing values in time series using ffill, bfill or interplate methods
def ts_impute(df, method):
    if method == "ffill":
        df = df.fillna(method="ffill")
    elif method == "bfill":
        df = df.fillna(method="bfill")
    elif method == "interpolate":
        df = df.interpolate()
    elif method == "knnimputer":
        imputer = KNNImputer()
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    elif isinstance(method, int) or isinstance(method, float):
        df = df.fillna(method)
    else:
        raise ValueError(f"Invalid method: {method}")
    return df


# SELECTORS


# Create a dataframe to store the anomalies
def anomalies_df(ts_teste, ts_mp, columns_ts, anomalies):
    # Create a dataframe to store the anomaly points
    anomaly_df = pd.DataFrame(columns=["Date", "Anomaly Value", "Column"])
    for column in columns_ts:
        i = columns_ts.index(column)
        for anomaly_index in anomalies[column]:
            anomaly_date = ts_teste.index[anomaly_index]
            anomaly_value = ts_mp[i][anomaly_index]
            new_row = pd.DataFrame(
                {
                    "Date": [anomaly_date],
                    "Anomaly Value": [anomaly_value],
                    "Column": [column],
                }
            )
            anomaly_df = pd.concat([anomaly_df, new_row], ignore_index=True)
    return anomaly_df


# Identify the anomalies according to the threshold
def threshold_anomalies(matrix_profile, columns_ts, threshold_factor=2):
    # Identify the anomalies in each dimension
    anomalies = {}
    thresholds = {}
    for i in range(matrix_profile.shape[0]):
        threshold = matrix_profile[i].mean() * threshold_factor
        thresholds[columns_ts[i]] = threshold
        anomalies[columns_ts[i]] = np.argwhere(matrix_profile[i] >= threshold).flatten()

    return anomalies, thresholds


# Identify the anomalies according to the top k values
def top_anomalies(matrix_profile, columns_ts, k=5):
    # Identify the anomalies in each dimension
    anomalies = {}
    for i in range(matrix_profile.shape[0]):
        # Identify the top k anomalous points based on the matrix profile values
        top_k_anomalies = np.argpartition(matrix_profile[i], -k)[-k:]
        anomalies[columns_ts[i]] = top_k_anomalies
    return anomalies


def df_row_selector(df, dict_conditions):
    """Selection of rows based on columns conditions"""
    c = dict_conditions.keys()
    row_mask = (
        df[c].isin(dict_conditions).all("columns")
    )  # note: df.isin returns all df columns, thus we need df[c]
    df = df.loc[row_mask, :].reset_index(drop=True)  # reset the index
    return df


# FEATURIZATION (feature engineering)


def z_score(df, column, affix="z"):
    zscore = ss.zscore(
        df[column], ddof=0
    )  # considering all population, then population standard deviation ddof=0
    df.insert(len(df.columns), u.code_name(affix, column), zscore.tolist())

    return df


def z_score_modified(df, column, consistency_correction=1.4826, affix="z_mod"):
    """
    Returns the modified z score and Median Absolute Deviation (MAD) from the scores in df.
    The consistency_correction factor converts the MAD to the standard deviation for a given
    distribution. The default value (1.4826) is the conversion factor if the underlying df
    is normally distributed
    """
    median = df[column].median()
    deviation_from_med = df[column] - median
    mad = deviation_from_med.abs().median()
    mod_zscore = deviation_from_med / (consistency_correction * mad)
    df.insert(len(df.columns), u.code_name(affix, column), mod_zscore.tolist())

    return df


# TIME SERIES PREPARATION / COMBINATIONS


# Prepare dataset with a timestamp if one does not exists format
# Calculate the number of possible combinations from the categorical features and the number of real combinations existing in the dataset
# TODO: Make it generalizable dependent on configs
def prepare_timestamp(df_preprocessed):
    # Create a Timestamp column from date and hour columns

    # Add Hour to start date
    df_preprocessed["Timestamp"] = df_preprocessed["Start Date"] + pd.to_timedelta(
        df_preprocessed["Start Hour"], unit="h"
    )

    # Format TimeStamp hour to 24h format (if we do this we must convert to datetime)
    df_preprocessed["Timestamp"] = df_preprocessed["Timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    df_preprocessed["Timestamp"] = pd.to_datetime(df_preprocessed["Timestamp"])

    # Drop Start Date and Start Hour columns
    df_preprocessed.drop(["Start Date", "Start Hour"], axis=1, inplace=True)

    # Create a list with the categorical columns as defined in the Json file before encoding the dataset
    # cols_cat=list(df.select_dtypes(include=['category']).columns)

    # Calculate the number of possibe combinations from the categorical features
    # print(f'Number of possible combinations: {df_preprocessed[cols_cat].nunique().prod()}')

    # Calculate the number of real combinations existing in the dataset
    # print(f'Number of existing combinations in the dataset: {df_preprocessed[cols_cat].drop_duplicates().shape[0]}')

    return df_preprocessed


# Create a dictionary with all time series for each combination of categorical features
def dict_time_series(
    dft, cols_cat, ts_name="Timestamp", cols_target=["#Events", "Event Duration"]
):
    # Set the Timestamp column as the DataFrame index
    dft = dft.set_index(ts_name)

    # Group the data by the categorical columns
    grouped = dft.groupby(cols_cat)

    # Create a dictionary to store the resulting time series
    time_series = {}

    # Iterate over the groups and create a time series for each group
    for name, group in grouped:
        # print(name)
        # Select the numerical columns
        ts_data = group[cols_target]

        # Create a time series for this group
        time_series[name] = ts_data  #

    return time_series


# Select the time series in dictionary (keys) where ts_column (Event Duration, #Events) have a minimum number of rows
def select_ts(ts_dict, ts_column, min_rows=100):
    # Select dataframes with the selected number of rows
    # selected_dfs = {k: v for k, v in ts_dict.items() if len(v) == max_rows}

    # Select only the keys of dataframes with the selected number of rows and Envent Duration sum > 0
    selected_dfs = [
        k for k, v in ts_dict.items() if len(v) >= min_rows and v[ts_column].sum() >= 0
    ]
    # selected_dfs = [k for k, v in ts_dict.items() if len(v) >= min_rows]
    #
    # print(f'Nº of dataframes selected: {len(selected_dfs)}')
    # sort ts1 by Event Duration descending

    # Select a random row from selected time series
    selected_df = selected_dfs[np.random.randint(0, len(selected_dfs))]

    print(f"Selected dataframe (combination of categorical features): {selected_df}")
    # print(f'Nº of dataframes selected: {len(selected_dfs)}')
    # display(selected_dfs)

    return ts_dict[selected_df]


# Calculate the window size to use in matrix profile. (This function should be improved!)
def calculate_window_size(frequency):
    freq = to_offset(frequency)
    if freq <= to_offset("1H"):
        window_size = 24
    elif freq > to_offset("1H") and freq <= to_offset("1D"):
        window_size = 7
    else:
        window_size = 3
    return window_size


# EVALUATION


"""Helper functions"""


def plot_comulative_probability(df, column):
    num_bins = 130
    normal_dist = [
        random.gauss(df[column].mean(), df[column].std()) for _ in range(1000)
    ]
    plt.figure()
    n, bins, patches = plt.hist(
        df[column],
        num_bins,
        density=True,
        histtype="step",
        cumulative=True,
        linewidth=1.0,
        label="df",
    )
    plt.hist(
        normal_dist,
        num_bins,
        density=True,
        histtype="step",
        cumulative=True,
        linewidth=1.0,
        label="normal distribution",
    )

    plt.grid(True)
    plt.legend(loc="upper left")
    axes = plt.gca()
    # axes.set_xlim([-400,400])
    plt.xlabel(f"{column}")
    plt.ylabel("Cumulative probability")
    plt.show()


# Plot the time series and the matrix profile
def plot_matrix_profile(
    ts_teste,
    ts_mp,
    matrix_profile,
    anomalies,
    window_size,
    columns_ts,
    freq,
    results_path,
    thresholds=None,
    save=False,
):
    # Check if we are calling from a interactive environment (notebook) or not
    plt = check_environment()

    # Set the figure size and spacing between subplots
    num_dim = matrix_profile.shape[0]
    fig_width = 10
    fig_height = num_dim * 6
    fig, axs = plt.subplots(num_dim * 2, 1, figsize=(fig_width, fig_height))
    plt.subplots_adjust(hspace=0.5)
    for i in range(num_dim):
        # Plot the time series
        axs[i * 2].plot(ts_teste.index, ts_mp[i])
        axs[i * 2].scatter(
            ts_teste.index[anomalies[columns_ts[i]]],
            ts_mp[i][anomalies[columns_ts[i]]],
            color="red",
        )
        axs[i * 2].set_title(f"Time Series for {columns_ts[i]}", fontsize=14)
        axs[i * 2].set_xlabel("Time")
        axs[i * 2].set_ylabel("Value")

        # Draw a rectangle and highlight the subsequence associated with each anomaly
        rect_width = pd.to_timedelta(window_size * freq)
        for anomaly_index in anomalies[columns_ts[i]]:
            rect = Rectangle(
                (ts_teste.index[anomaly_index], np.min(ts_mp[i])),
                rect_width,
                np.ptp(ts_mp[i]),
                fill=False,
                edgecolor="green",
            )
            # axs[i * 2].add_patch(rect)
            axs[i * 2].axvspan(
                ts_teste.index[anomaly_index],
                ts_teste.index[anomaly_index] + rect_width,
                alpha=0.2,
                color="yellow",
            )

        # Plot the matrix profile
        axs[i * 2 + 1].plot(
            ts_teste.index[: -window_size + 1], matrix_profile[i], color="orange"
        )
        axs[i * 2 + 1].scatter(
            ts_teste.index[: -window_size + 1][anomalies[columns_ts[i]]],
            matrix_profile[i][anomalies[columns_ts[i]]],
            color="red",
        )
        if thresholds is not None:
            axs[i * 2 + 1].axhline(
                y=thresholds[columns_ts[i]],
                color="green",
                linestyle="--",
                label="Threshold",
            )
            axs[i * 2 + 1].legend()
            chart_filename = results_path + "Matrix_Profile_Threshold.png"
        else:
            chart_filename = results_path + "Matrix_Profile_TopN.png"
        # axs[i * 2 + 1].axhline(y=matrix_profile[i].mean(), color='silver', linestyle='--',label='Mean MP distance')
        # axs[i * 2 + 1].legend()
        axs[i * 2 + 1].set_title(f"Matrix Profile for {columns_ts[i]}", fontsize=14)
        axs[i * 2 + 1].set_xlabel("Time")
        axs[i * 2 + 1].set_ylabel("Matrix Profile Value")

        # Ensure that the x-axis of both charts match
        axs[i * 2 + 1].set_xlim(axs[i * 2].get_xlim())

    # Save figure on specified path
    if save:
        plt.savefig(chart_filename)

    plt.show()

    return


# Create a chart zooming on anomaly points
def plot_zoomed_anomalies(
    ts_teste, ts_mp, anomalies, window_size, columns_ts, results_path, zoom_type
):
    # Check if we are calling from a interactive environment (notebook) or not
    plt = check_environment()

    zoom_factor = 10
    anomaly_charts = []
    for i in range(len(columns_ts)):
        packed_anomalies = []
        for j, anomaly_index in enumerate(anomalies[columns_ts[i]]):
            start_index = max(0, anomaly_index - window_size * zoom_factor)
            end_index = min(
                len(ts_teste) - window_size + 1,
                anomaly_index + window_size * zoom_factor,
            )
            x_range = range(start_index, end_index)
            filtered_anomalies = [
                x for x in anomalies[columns_ts[i]] if start_index <= x < end_index
            ]
            filtered_anomalies_indices = [x - start_index for x in filtered_anomalies]
            if set(filtered_anomalies).issubset(set(packed_anomalies)):
                continue
            packed_anomalies.extend(filtered_anomalies)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(ts_teste.index[x_range], ts_mp[i][x_range])
            ax.scatter(
                ts_teste.index[x_range][filtered_anomalies_indices],
                ts_mp[i][x_range][filtered_anomalies_indices],
                color="red",
            )
            ax.set_title(f"Zoomed on anomalies for {columns_ts[i]}", fontsize=14)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            anomaly_charts.append(fig)

            # Save figure on specified path
            chart_filename = (
                results_path + zoom_type + columns_ts[i] + "_" + str(j) + ".png"
            )
            plt.savefig(chart_filename)
            plt.close(fig)

    # chart_filename = results_path + zoom_type + ".png"
    # plt.savefig(chart_filename)

    plt.show()

    return anomaly_charts


def check_environment():
    # Check if we are calling from a notebook or not
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            matplotlib.use("Agg")
    except (AttributeError, ImportError):
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt
