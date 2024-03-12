from pathlib import Path
import os
from IPython.display import display
from sklearn.pipeline import Pipeline
import pandas as pd


# prototype imports
import prototype.utils as u
import prototype.datasets as d
import prototype.models as m

# package pipeline_functions imports
import prototype.pipelines.pipeline_functions as pipe

"""Pipeline functions"""


def configurations(data_path, config_path):
    # Configurations
    config_dataset, config_pipeline = pipe.configurations(data_path, config_path)

    return config_dataset, config_pipeline


def load(config_dataset, config_pipeline, check=False):
    print("\nload dataset:\n")
    # Read the dataset
    df, cols_cat, cols_num, cols_target = pipe.load(config_dataset, config_pipeline)
    # check dataframe
    if check:
        pipe.check_dataframe(df)

    return df, cols_cat, cols_num, cols_target


def preprocessing(df, config_pipeline, preprocessing_pipe, cols_cat, cols_num, cols_target, check=False):
    print("\npreprocessed dataset:\n")
    results = None
    # custom pipeline: (if we want to change the default pipeline steps, we can do it here - remove or add steps to the pipeline list)
    pipe_list = [
        ("convert_data_types", preprocessing_pipe.convert_data_types),
        ("drop_duplicates", preprocessing_pipe.drop_duplicates),
        ("imputing_num", preprocessing_pipe.data_imputing_num),
    ]
    preprocessing_pipe.pipeline = Pipeline(pipe_list)  # update default pipeline
    results = pipe.preprocessing(df, config_pipeline, preprocessing_pipe, cols_cat, cols_num, cols_target)
    if check:
        pipe.check_dataframe(results)

    return results


def models(check=False):
    print("\nselected models:\n")
    model_d = pipe.models()  # default dict
    model_d = {
        "Linear Regression": m.LinRegression(),
        "Linear Regression 2": m.LinRegression(),
    }  # custom dict

    if check:
        display(model_d)
    return model_d


def train(X, y, model, model_name):
    results = None
    # custom pipeline:
    pipe_list = [
        ("remove_nan_columns", preprocessing_pipe.remove_nan_columns),
        ("scale_data", preprocessing_pipe.scale_data),
        ("tranform_inputs", preprocessing_pipe.tranform_inputs),
        ("split_data", preprocessing_pipe.split_data),
        ("train_test_models", preprocessing_pipe.train_test_models),
    ]
    preprocessing_pipe.preprocessing_pipeline = Pipeline(pipe_list)
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
    preprocessing_pipe = PreprocessingPipe(config_pipeline, cols_cat, cols_num, cols_target)
    df_t = preprocessing(
        df,
        config_pipeline,
        preprocessing_pipe,
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
