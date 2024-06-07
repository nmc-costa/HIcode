from sklearn.pipeline import Pipeline

# prototype imports
import package.preprocessing.preprocessing as pp


class PreprocessingPipe:
    def __init__(self, config, cols_cat, cols_num, cols_target):
        self.config = config
        self.cols_cat = cols_cat
        self.cols_num = cols_num
        self.cols_target = cols_target
        self.instatiate_classes()
        self.pipeline_list()

    def instatiate_classes(self):
        # 1. INSTANTIATE THE Preprocessing CLASS's
        ## Convert data types according datacontract before encoding dataset to analyze it (convert fields to categorical and manage corrupted data in numerical fields)
        self.convert_data_types = pp.ConvertDataTypes(numerical_cols=self.cols_num, categorical_cols=self.cols_cat)
        self.drop_duplicates = pp.DropDuplicates()
        self.data_imputing_num = pp.DataImputing(strategy="mean", impute_cols=self.cols_num)

    def pipeline_list(self):
        # 2. DEFINE PIPELINE
        ## Apply a pipeline with or without scaling. If we split the dataset in train and test we must not scale the whole dataset. We must scale the train dataset and then apply the same transformation to the test dataset)
        pipe_list = [
            ("convert_data_types", self.convert_data_types),
            ("drop_duplicates", self.drop_duplicates),
            ("imputing_num", self.data_imputing_num),
        ]
        self.pipeline = Pipeline(pipe_list)
