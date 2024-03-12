from pathlib import Path
import os
from IPython.display import display
from sklearn.pipeline import Pipeline
import pandas as pd


# prototype imports
import prototype.utils as u
import prototype.datasets as d
import prototype.models as m

# config imports
from prototype.cfg_preprocessing_pipe import PreprocessingPipe

"""Pipeline functions"""


def configurations(data_path, config_path):
    # Configurations
    config_dataset = d.read_config(data_path)  # Read the data contract Json file
    config_pipeline = d.read_config(config_path)  # Read the configuration Json file

    return config_dataset, config_pipeline


def load(config_dataset, config_pipeline):
    # Load dataset
    dataset_path = (
        Path(os.path.dirname(os.path.abspath(__file__))) / Path(config_pipeline["dataset_location"]) / Path(config_pipeline["dataset_name"])
    )  # solve absolute and relative paths (warning: relative path should be considered from this __file__)
    # Read the dataset
    df, cols_cat, cols_num, cols_target = d.read_dataset(
        config_dataset,
        dataset_path=dataset_path,
        separator=config_pipeline["separator"],
    )

    return df, cols_cat, cols_num, cols_target


def preprocessing(df, config_pipeline, preprocessing_pipe, cols_cat, cols_num, cols_target):
    # Transform Pipeline
    df_t = preprocessing_pipe.pipeline.fit_transform(df)
    # display(df_t)

    return df_t


def models():  # default model is Linear Regression
    model_d = {"Linear Regression": m.LinRegression()}
    return model_d


def train(X, y, model, model_name):
    model.fit(X, y)
    y_pred = model.predict(X)
    results = pd.DataFrame({"y": y, "y_pred": y_pred, "model_name": model_name})
    results.plot()
    return results


"""Helper functions"""


def check_dataframe(df, method="pandas"):
    if method == "pandas":
        display(df.head(5))
        print("shape: ", df.shape)
    elif method == "pyspark":
        pass

    return


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
    config_dataset, config_pipeline = configurations(data_path, config_path)

    # Load dataset
    df, cols_cat, cols_num, cols_target = load(config_dataset, config_pipeline)

    # Transform dataset
    df_t = transform(
        df,
        config_pipeline,
        cols_cat,
        cols_num,
        cols_target,
    )  # Read the dataset and preprocess it

    # Train models
    model_d = models()
    X = df_t[cols_target[1:]].to_numpy()
    y = df_t[cols_target[0]].to_numpy()
    results_d = {}
    for model_name, model in model_d.items():
        print(f"Training {model_name}")
        results = train(X, y, model, model_name)
        results_d[model_name] = results

    # Save results
    if save:
        u.save_csv(df_t, config_pipeline["output_results"], config_pipeline["dataset_name"])

    return df_t, results_d


if __name__ == "__main__":
    cwd = Path(os.path.dirname(os.path.abspath(__file__)))  # Get the current working directory
    pipeline(cwd / "cfg_pipeline.json", cwd / "cfg_dataset.json", save=False)
