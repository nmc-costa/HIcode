"""Utils Methods"""
# system
import os
from pathlib import Path


"""Save functions"""


def save_csv(df, folder_path, name):
    # extract .csv from the file name
    file_name = name.split(".")[0] + "_preprocessed" + ".csv"
    folder_path = Path(
        folder_path
    ).resolve()  # resolve relative paths inside developer package

    # Check if the folder exists and create it if it doesn't
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the DataFrame as a CSV file in the specified folder
    df.to_csv(os.path.join(folder_path, file_name), index=False)

    return
