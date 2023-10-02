"""Utils Methods"""
# system
import os
from pathlib import Path
#
import json

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


"""Save functions"""


def save_csv(df, folder_path, name):
    # extract .csv from the file name
    file_name = name.split(".")[0] + "_preprocessed" + ".csv"
    folder_path = Path(folder_path).resolve() # resolve relative paths inside developer package
    
    # Check if the folder exists and create it if it doesn't
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the DataFrame as a CSV file in the specified folder
    df.to_csv(os.path.join(folder_path, file_name), index=False)

    return
