# Pipeline: example of transforming experiments.e1.ipynb into an HIcode pipeline application

This Python project implements a pipeline for data ingestion. The pipeline comprises the following files:
* Pipeline metadata: cfg_pipeline.json
* Dataset metadata: cfg_dataset.json
* Transformations metadata: cfg_transform_pipe.py
* Python scripts: p1_pipeline.py, preprocessing.py, datasets.py, utils.py
* Python notebook: p1_run.ipynb

## Pipeline:
- Configurations
- Load dataset
- Transform dataset
- Train models
- Save results


## Run the application

After defining the configuration parameters in the 'cfg_dataset', 'cfg_pipeline' JSON files, the application can be run by executing the pipeline function defined in p1_pipeline.py as this (see in notebook):
````python
    pipeline(
            'cfg_pipeline.json',
            'cfg_dataset.json',
        )
