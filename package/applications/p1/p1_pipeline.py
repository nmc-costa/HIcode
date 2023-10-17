from pathlib import Path
import os
from IPython.display import display
from sklearn.pipeline import Pipeline
import pandas as pd


# package imports
import package.utils.utils as u
import package.datasets.datasets as d
import package.models.models as m

# config imports
from package.preprocessing.cfg_transform_pipe import TransformPipe

"""Pipeline functions"""


def configurations(data_path, config_path):
    # Configurations
    config_dataset = d.read_config(data_path)  # Read the data contract Json file
    config_pipeline = d.read_config(config_path)  # Read the configuration Json file

    return config_dataset, config_pipeline  # Return


def load(config_dataset, config_pipeline):
    # Load dataset

    dataset_path = Path(config_pipeline["dataset_location"]) / Path(
        config_pipeline["dataset_name"]
    )
    # resolve relative paths inside developer package
    dataset_path = dataset_path.resolve()
    # Read the dataset
    df, cols_cat, cols_num, cols_target = d.read_dataset(
        config_dataset,
        dataset_path=dataset_path,
        separator=config_pipeline["separator"],
    )

    return df, cols_cat, cols_num, cols_target


def transform(df, config_pipeline, cols_cat, cols_num, cols_target):
    # Transform Pipeline
    transform_pipe = TransformPipe(config_pipeline, cols_cat, cols_num, cols_target)
    preprocessing_pipeline = Pipeline(transform_pipe.pipe_list)
    df_t = preprocessing_pipeline.fit_transform(df)
    display(df_t)

    return df_t


def models():
    model_d = {"Linear Regression": m.LinRegression()}
    return model_d


def train(X, y, model, model_name):
    model.fit(X, y)
    y_pred = model.predict(X)
    results = pd.DataFrame({"y": y, "y_pred": y_pred, "model_name": model_name})
    results.plot()
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
        u.save_csv(
            df_t, config_pipeline["output_results"], config_pipeline["dataset_name"]
        )

    return df_t, results_d


if __name__ == "__main__":
    cwd = Path(
        os.path.dirname(os.path.abspath(__file__))
    )  # Get the current working directory
    pipeline(
        cwd / "cfg_pipeline.json", cwd / "../../datasets/cfg_dataset.json", save=False
    )
