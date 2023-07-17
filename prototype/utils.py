# system
from pathlib import Path
import platform
import sys
import os

#
import json
import numpy as np
import matplotlib.pyplot as plt


"""Init rules"""


def code_name(new_affix, old_name):
    return f"{new_affix}_{old_name}"


"""Data Contract functions"""


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


def save_config(config, filename):
    """Saves the configuration"""
    with open(filename, "w") as dump_config:
        dump_config.write(json.dumps(config))


# Get the list of features in the dataset according to the data type
def get_fields_by_datatype(data, data_type, exclude_vars):
    result = []
    for key in data:
        for field in data[key]:
            if (field["data_type"].upper().find(data_type.upper()) > -1) and (
                field["name"] not in exclude_vars
            ):
                result.append(field["name"])
    return result


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


# create a list with the name of excluded variables
def get_excluded_vars(data):
    exclude_vars = get_list_vars(data, "excluded_variables")
    return exclude_vars


# create a list with the name of target variables
def get_target_vars(data):
    target_vars = get_list_vars(data, "target_variables")
    return target_vars


# Format timestamp according to the timestamp format (it converts the field to object)
def format_timestamp(df, data):
    for variable in data["time_name"]:
        if variable["data_type"] == "Date":
            print(variable["timestamp_format"])
            df[variable["name"]] = df[variable["name"]].apply(
                lambda x: x.strftime(variable["timestamp_format"])
            )
    return df


"""Save functions"""


def save_csv(df, folder_path, name):
    import os

    # extract .csv from the file name
    file_name = name.split(".")[0] + "_preprocessed" + ".csv"

    # Check if the folder exists and create it if it doesn't
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the DataFrame as a CSV file in the specified folder
    df.to_csv(os.path.join(folder_path, file_name), index=False)

    return


"""Path functions"""


def parse_paths(init_config):
    """parse_paths"""
    # Define root dir dependent on OS
    rdir = Path(init_config["rdir"])
    dataset = Path(init_config["dataset"])
    ddir = Path(init_config["ddir"])
    outdir = Path(init_config["outdir"])
    if (
        str(platform.platform()).find("Linux") > -1
    ):  # if using windows and linux one can change paths
        rdir = rdir
        outdir = outdir
    print(f"OS: {platform.platform()}")
    print(f"package root dir: {rdir}")
    print(f"dataset dir: {ddir}")
    print(f"output dir: {outdir}")

    return rdir, dataset, ddir, outdir


def assure_path_exists(path):
    """
    Create directory folders from path

    Note: windows path strings should be r'path'
    """

    # Only for dirs - for complete you have to change dir for path
    # dirname is obrigatory - make sure it is a dir
    dir_p = os.path.normpath(path)
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)


"""Helper functions"""
