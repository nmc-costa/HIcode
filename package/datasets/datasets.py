"""Datasets Methods"""
import json
import pandas as pd


"""Dataset Config functions"""


# Read the JSON configuration file
def read_config(json_file):
    try:
        # Read the JSON file
        with open(json_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: {json_file} not found")
    except json.JSONDecodeError:
        print(f"Error: {json_file} is not a valid JSON file")


# Get the list of features in the dataset according to the data type
def get_fields_by_keyvalue(data, keyvalue=["data_type", "Date"], exclude_vars=[]):
    result = []
    for key in data:
        for field in data[key]:
            try:
                if (field[keyvalue[0]].upper().find(keyvalue[1].upper()) > -1) and (
                    field["name"] not in exclude_vars
                ):
                    result.append(field["name"])
            except Exception as e:
                continue
    return result


# Get a dict of date formats
def get_date_formats(data, date_vars, exclude_vars):
    result = {}
    for key in data:
        for field in data[key]:
            if (field["name"] in date_vars) and (field["name"] not in exclude_vars):
                result.update({field["name"]: field["timestamp_format"]})

    return result


# create a list with the name of variables
def get_list_vars(data, list_name):
    target_vars = []
    for variable in data[list_name]:
        target_var = variable["name"]
        target_vars.append(target_var)
    return target_vars


"""Dataset read functions"""


# READ/LOAD DATASET
## read the dataset according dataContract
def read_dataset(
    data_contract, dataset_path="", separator=",", data_type_key="data_type"
):
    # Identify datatypes and excluded variables
    exclude_vars = get_list_vars(data_contract, "excluded_variables")
    str_vars = get_fields_by_keyvalue(
        data_contract,
        keyvalue=[data_type_key, "str"],
        exclude_vars=exclude_vars,
    )
    int_vars = get_fields_by_keyvalue(
        data_contract,
        keyvalue=[data_type_key, "int"],
        exclude_vars=exclude_vars,
    )
    float_vars = get_fields_by_keyvalue(
        data_contract,
        keyvalue=[data_type_key, "float"],
        exclude_vars=exclude_vars,
    )
    cat_vars = get_list_vars(data_contract, "categorical_variables")
    cat_vars += [value for value in str_vars if value not in cat_vars]
    num_vars = int_vars + float_vars
    date_vars = get_fields_by_keyvalue(
        data_contract,
        keyvalue=[data_type_key, "Date"],
        exclude_vars=exclude_vars,
    )
    date_formats = get_date_formats(data_contract, date_vars, exclude_vars)
    target_vars = get_list_vars(data_contract, "target_variables")

    # Read csv file into a dataframe specifying separator and dtypes for each column
    df = pd.read_csv(
        dataset_path,
        sep=separator,
        usecols=lambda x: x not in exclude_vars,
    )

    # Parse the date columns using pd.to_datetime() with format parameter
    # NOTE: doing after read_csv due to deprepactions from pandas v 2.0.0 date_vars and date_formats can be used in pd.read_csv
    for column, date_format in date_formats.items():
        df[column] = pd.to_datetime(df[column], format=date_format)

    return df, cat_vars, num_vars, target_vars
