from pathlib import Path
import os
from sklearn.pipeline import Pipeline
import pandas as pd


# prototype imports
import prototype.utils as u
import prototype.datasets as d
import prototype.models as m

# config imports
from prototype.pipelines.p1.cfg_transform_pipe import TransformPipe

"""Pipeline functions"""


def configurations(data_path, config_path):
    # Configurations
    data_contract = d.read_config(data_path)  # Read the data contract Json file
    config = d.read_config(config_path)  # Read the configuration Json file

    return data_contract, config  # Return


def load(data_contract, config):
    # Load dataset

    dataset_path = Path(config["dataset_location"]) / Path(config["dataset_name"])
    # resolve relative paths inside developer package (WARNING: on notebook should change the workspace)
    dataset_path = dataset_path.resolve()
    # Read the dataset
    df, cols_cat, cols_num, cols_target = d.read_dataset(
        data_contract, dataset_path=dataset_path, separator=config["separator"]
    )

    return df, cols_cat, cols_num, cols_target


def transform(df, config, cols_cat, cols_num, cols_target):
    # Transform Pipeline
    transform_pipe = TransformPipe(config, cols_cat, cols_num, cols_target)
    preprocessing_pipeline = Pipeline(transform_pipe.pipe_list)
    df_t = preprocessing_pipeline.fit_transform(df)

    return df_t


def models():
    model_d = {
        "Linear Regression": m.LinRegression(),
        "Ridge Regression (0.5)": m.RidgeRegression(0.5),
    }
    return model_d


def train(X, y, model, model_name):
    model.fit(X, y)
    y_pred = model.predict(X)
    results = pd.DataFrame({"y": y, "y_pred": y_pred, "model_name": model_name})
    return results


"""Helper functions"""


"""Pipeline"""


def pipeline(config_path, data_path, save=False):
    """
    - Configurations
    - Load dataset
    - Transform dataset
    - Train models
    - Save results
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

    # Train models
    model_d = models()
    X = df_t[cols_target[1:]].to_numpy()
    y = df_t[cols_target[0]].to_numpy()
    for model_name, model in model_d.items():
        print(f"Training {model_name}")
        results = train(X, y, model, model_name)
        results.plot()

    # Save results
    if save:
        u.save_csv(df_t, config["output_results"], config["dataset_name"])

    return df_t


if __name__ == "__main__":
    cwd = Path(
        os.path.dirname(os.path.abspath(__file__))
    )  # Get the current working directory
    pipeline(cwd / "cfg_pipeline.json", cwd / "cfg_dataset.json", save=False)
