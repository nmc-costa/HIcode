# HIcode guidelines (v1.0)
<img src="./docs/Slide2.PNG">


**Prototype (DS)**:

*experiments folder (individual)*:
The 1st stage of code (the lab). Use it for experimenting with anything you see fit. Put it in a way that can be tested by anyone. The idea is to simplify as much as you can so that you find functions that can be reused (reusable). In later iterations, it should have a similar architecture as the pipelines folder.

*root (team)*:
The 2nd stage after refining the experiments, reusable functions should be sent to the root/source scripts. The root is a team effort and should follow the Integration Wrike and GIT (“feature/ ProjectName-WrikeID-short-description”).

- **utils.py**: Contains generic functions/classes and initial coding rules that can be reused (e.g., paths, code column names, etc).
- **preprocessing.py**: Includes all the functions/classes that are used to preprocess/prepare the datasets.
- **model.py**: Encompasses all the functions/classes that are needed to create a model and support model visualizations, etc. (reusable functions).
- **train.py**: Comprises all the functions/classes for training routines.
- **monitor.py**: Contains all the functions/classes for monitoring the models.

*pipelines folder (team)*:
The 3rd stage, it should be used to create full pipelines with config files, the git already has 2 examples. It will be used for doing a pipeline for autoencoders, for monitoring, etc. Each pipeline should demonstrate the functionality of major features that have sub-features.

*tutorial notebook (team)*:
The last stage, for us to show the complete functionalities of the package in the most simplified way, in this, we can call the pipelines and show the most important results.

*configs folder (team)*:
This folder will store the final config files from pipelines.

*data_sample folder*:
An original sample of data that is being used to test all the prototypes.

**Package (DE)**:

*Application folder (team)*:
1st stage import pipelines as applications and use the prototype as the library.

*Final stage*:
Copy and paste prototype root/source scripts inside the package and change relative imports to finalize the package.
