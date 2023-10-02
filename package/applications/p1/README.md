# Pipeline: example of transforming experiments.e1.ipynb into an HIcode pipeline 

This Python project implements a pipeline for data ingestion. The pipeline comprises the following files:
* Dataset metadata: cfg_data_contract.json
* Package metadata: cfg_init_vars.json
* Transformations metadata: cfg_transform_pipe.py
* Python scripts: e1_pipe.py, preprocessing.py, utils.py
* Python notebook: e1_run.ipynb

## Preprocessing data
The following classes are available for preprocessing data and are configured in 'cfg_transform_pipe.py':
* ConvertDataTypes: Convert data types and correct corrupted characters in numerical fields
* DropDuplicate: Drop duplicated rows

## Run the application

After defining the configuration parameters in the 'data_contract', 'cfg_init_vars' JSON files, the application can be run by executing the pipeline function defined in e1_pipe.py as this (see in notebook):
````python
    pipeline(
            'cfg_init_vars.json',
            'cfg_data_contract.json',
        )
