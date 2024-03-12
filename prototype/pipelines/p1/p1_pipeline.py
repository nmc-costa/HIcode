from pathlib import Path
import os
from IPython.display import display
from sklearn.pipeline import Pipeline
import pandas as pd


# package imports
import prototype.utils as u
import prototype.datasets as d
import prototype.models as m

# package config imports
from prototype.cfg_preprocessing_pipe import PreprocessingPipe

# package pipeline_functions imports
import prototype.pipelines.pipeline_functions as pipe

"""Pipeline functions"""


def configurations(data_path, config_path):
    # Configurations
    config_dataset, config_pipeline = pipe.configurations(data_path, config_path)

    return config_dataset, config_pipeline


def load(config_dataset, config_pipeline, check=False):
    print("\nload dataset:\n")
    df, cols_cat, cols_num, cols_target = pipe.load(config_dataset, config_pipeline)

    # check dataframe
    if check:
        pipe.check_dataframe(df)

    return df, cols_cat, cols_num, cols_target


def preprocessing(df, preprocessing_pipe, check=False):
    print("\npreprocessed dataset:\n")
    results = None
    # NOTE: custom pipeline: if we want to change the default pipeline steps, we can do it here - remove or add steps to the pipeline list
    # NOTE: test this by removing steps
    pipe_list = [
        ("convert_data_types", preprocessing_pipe.convert_data_types),
        ("drop_duplicates", preprocessing_pipe.drop_duplicates),
        ("imputing_num", preprocessing_pipe.data_imputing_num),
    ]
    preprocessing_pipe.pipeline = Pipeline(pipe_list)  # update default pipeline
    results = pipe.preprocessing(df, preprocessing_pipe)

    if check:
        pipe.check_dataframe(results)

    return results


def models(check=False):
    print("\nselected models:\n")
    model_d = pipe.models()  # default dict
    # NOTE: custom dict: if we want to change the default models, we can do it here - remove or add models to the model_d dictionary
    model_d = {
        "Linear Regression": m.LinRegression(),
        "Linear Regression 2": m.LinRegression(),
    }

    if check:
        display(model_d)
    return model_d


def multi_train(X, y, model_d, check=False):
    print("\ntrain multi models:\n")
    results_d = pipe.multi_train(X, y, model_d, check=check)

    if check:
        pass

    return results_d


"""Helper functions"""


"""Pipeline"""


def pipeline(config_path, data_path, save=False, check=False):
    """
    - Configurations
    - Load dataset
    - Transform dataset
    - Train/test models
    - Save results
    """
    # Configurations
    config_dataset, config_pipeline = configurations(data_path, config_path)

    # Load dataset
    df, cols_cat, cols_num, cols_target = load(config_dataset, config_pipeline, check=check)

    # Transform dataset
    preprocessing_pipe = PreprocessingPipe(config_pipeline, cols_cat, cols_num, cols_target)
    df_t = preprocessing(
        df,
        preprocessing_pipe,
        check=check,
    )  # Read the dataset and preprocess it

    # Train/test models
    """
    # NOTE: Like preprocessing.py the train.py file should be a collection of classes that are used to train the model.
    # NOTE: DO THE FOLLOWING EXERCISE TO CHECK WHAT YOU LEARNED: 
    # # 1. pass the next code to a class in the train.py file like in the preprocessing.py file 
    # # 2. create cfg_train_pipe.py like cfg_preprocessing_pipe.py
    # # 3. update a train() and multi_train() functions in pipeline_functions.py file 
    """
    model_d = models(check=check)
    X = df_t[cols_target[1:]].to_numpy()
    y = df_t[cols_target[0]].to_numpy()
    results_d = multi_train(X, y, model_d, check=check)

    # Save results
    if save:
        u.save_csv(df_t, config_pipeline["output_results"], config_pipeline["dataset_name"])

    return df_t, results_d


if __name__ == "__main__":
    cwd = Path(os.path.dirname(os.path.abspath(__file__)))  # Get the current working directory
    pipeline(cwd / "cfg_pipeline.json", cwd / "../../cfg_dataset.json", check=True)
