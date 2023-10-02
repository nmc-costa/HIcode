from pathlib import Path
import os
from sklearn.pipeline import Pipeline


# package imports
import package.utils.utils as u
import package.preprocessing.preprocessing as pp

# config imports
from .cfg_transform_pipe import TransformPipe

"""Pipeline functions"""


def configurations(data_path, config_path):
    # Configurations
    data_contract = u.read_config(data_path)  # Read the data contract Json file
    config = u.read_config(config_path)  # Read the configuration Json file

    return data_contract, config  # Return


def load(data_contract, config):
    # Load dataset

    dataset_path = Path(config["dataset_location"]) / Path(config["dataset_name"])
    # resolve relative paths inside developer package (WARNING: on notebook should change the workspace)
    dataset_path = dataset_path.resolve()
    # Read the dataset
    df, cols_cat, cols_num, cols_target = pp.read_dataset(
        data_contract, dataset_path=dataset_path, separator=config["separator"]
    )

    return df, cols_cat, cols_num, cols_target


def transform(df, config, cols_cat, cols_num, cols_target):
    # Transform Pipeline
    transform_pipe = TransformPipe(config, cols_cat, cols_num, cols_target)
    preprocessing_pipeline = Pipeline(transform_pipe.pipe_list)
    df_t = preprocessing_pipeline.fit_transform(df)

    return df_t


def model():
    pass


def train():
    pass


"""Helper functions"""


"""Pipeline"""


def pipeline(config_path, data_path, save=False):
    """
    - Configurations
    - Load dataset
    - Transform dataset
    """
    # Configurations
    data_contract, config = configurations(data_path, config_path)

    # Load dataset
    df, cols_cat, cols_num, cols_target = load(data_contract, config)

    # Transform dataset
    df_t = transform(
        df,
        config,
        cols_cat,
        cols_num,
        cols_target,
    )  # Read the dataset and preprocess it

    if save:
        u.save_csv(df_t, config["output_results"], config["dataset_name"])

    return df_t


if __name__ == "__main__":
    cwd = Path(
        os.path.dirname(os.path.abspath(__file__))
    )  # Get the current working directory
    pipeline(cwd / "cfg_init_vars.json", cwd / "cfg_data_contract.json", save=False)
