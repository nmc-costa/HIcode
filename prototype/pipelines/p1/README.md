# Data Ingestion and data profiling

This Python project implements a pipeline for data ingestion and profiling. The pipeline comprises the following files:
* Dataset metadata: DataContract.json
* Package metadata: config_prep.json
* Transformations metadata: config_transform_pipe.py
* Python scripts: e1_pipe.py, preprocessing.py, utils.py
* Python notebook: e1_profiling.ipybn

## Preprocessing data
The following classes are available for preprocessing data:
* ConvertDataTypes: Convert data types and correct corrupted characters in numerical fields
* DropDuplicate: Drop duplicated rows
* DropNaN: Drop rows and columns above certain threshold defined in 'DataContract'
* DataImputing: Impute missing values according one of the possible values ('mean', 'median', 'most frequent') 
* DataEncoding: Encode categorical variables according one of the possible values (label, onehot) for label or onehot encoding
* DataScaling: Scale numerical data according to one of the possible values (standard, minmax) to use StandarScaler or MinMaxScaler methods 

## Profiling
* Displays statistical information regarding original and transformed dataset after preprocessing data
* Uses DataPrep to create a html report with detailed information about each dataset.

## Run the application

After defining the configuration parameters in the 'DataContract', 'config_prep' JSON files, the application can be run by executing the pipeline function defined in e1_pipe.py as this (see in notebook):
````python
    pipeline(
            'config_prep.json',
            'dataContract.json',
        )
