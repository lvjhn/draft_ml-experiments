
# =========================================================================== #
# :: DATASET CLASS
#
# :: Description ::
#
# Defines a dataset class which loads data from the datasets folder and reports 
# its data.
#
# =========================================================================== #

# ==== COMMON IMPORTS ======================================================= #
from setup.settings import *
from core.helpers.paths import *
from core.helpers.common import *
from core.helpers.wrappers import * 
from core.helpers.presets import * 
# =========================================================================== #

# ==== NATIVE IMPORTS ======================================================= #
from .dataset import Dataset
# =========================================================================== #

# ==== PROJECT IMPORTS ====================================================== #
from core.helpers.classifiers import *
from core.helpers.regressors import *
from core.helpers.common_imports import * 

from core.helpers.presets import *
# =========================================================================== #

# ==== MAIN SCRIPT ========================================================== # 

class LabeledDataset: 
    def __init__(
        self, 
        dataset = None,
        label = None, 
        feature_mode = "include",
        task_type = "classification",
        features = None,
        remove_ids = True,
        holdout = 0.3,
        verbose = True,
        indent = ""
    ) :    
        """ 
            Labels a dataset using a specified label column and provides
            the option to select features automatically or manually. 

            Parameters: 
                dataset      
                    - the dataset to assigned to this object 

                label        
                    - the label column to use, default uses the last
                      column

                feature_mode 
                    - whether "include" or "exclude" the columns 
                      in `features` array
                    
                features     
                    - the array of features for this labeled dataset either 
                      to be excluded or included
                    
                remove_ids   
                    - whether to remove id columns (default is true)

                holdout 
                    - holdout fraction from the entire dataset
        """ 

        # messages #
        self.verbose = verbose 
        self.indent = indent 

        # main parameters # 
        self.params = {
            "dataset" : dataset,
            "label" : label, 
            "feature_mode" : feature_mode,
            "features" : features,
            "remove_ids" : remove_ids,
            "task_type" : task_type,
            "holdout" : holdout
        } 
        
        # state objects # 
        self.state = {
            "dataset" : dataset,
            "feature_columns" : [],
            "label_column" : None,
            "X"       : None, 
            "y"       : None 
        }

        self.X = self.state["X"]
        self.y = self.state["y"]

        # flow
        self.define_features_and_labels()

        if remove_ids: 
            self.remove_ids()
        
        self.extract_features_and_labels()
   
    ###########################################################################

    def define_features_and_labels(self): 
        feature_columns_param = self.params["features"] 
        label_column_param = self.params["label"]

        feature_mode = self.params["feature_mode"]
        all_columns = list(self.state["dataset"].columns())

        feature_columns = []
        label_column = None 

        # if no feature columns argument is passed 
        # assign all columns as feature columns
        if feature_columns_param is None: 
            feature_columns = all_columns 

        # if feature columns are passed, determine 
        # columns based on inclusion or exlusion rule 
        # in feature_mode
        else:
            if feature_mode == "include":
                feature_columns = feature_columns_param 
            elif feature_mode == "exclude": 
                feature_columns = all_columns 
                for column in feature_columns_param: 
                    if column in feature_columns:
                        feature_columns.remove(column)
            else:
                raise Exception(f"Unknown feature mode [{feature_mode}]")

        # if no label column is passed 
        # assigned it to the last column by default 
        if label_column_param == None: 
            label_column = all_columns[-1] 

        # if a label column was passed assign it as 
        # the label column 
        else: 
            label_column = label_column_param 

        # remove the label column from the features list if 
        # it exists in such set 
        if label_column in feature_columns: 
            feature_columns.remove(label_column) 

        self.state["feature_columns"] = feature_columns 
        self.state["label_column"] = label_column

    ###########################################################################
    
    def remove_ids(self): 
        feature_columns = self.state["feature_columns"]
        dataset = self.state["dataset"]

        for column in feature_columns: 
            type_ = dataset.state["types"][column]
            if type_ == "id": 
                feature_columns.remove(column)

        return feature_columns

    #############################################################################

    def extract_features_and_labels(self): 
        feature_columns = self.state["feature_columns"]
        label_column = self.state["label_column"]

        df = self.state["dataset"].state["df"]

        self.state["AX"] = df[feature_columns]
        self.state["Ay"] = df[label_column]

        if self.params["holdout"] != None:
            if self.params["task_type"] == "classification":
                self.state["X"], self.state["HX"], \
                self.state["y"], self.state["Hy"], = \
                    train_test_split(
                        self.state["AX"],
                        self.state["Ay"],
                        stratify=self.state["Ay"],
                        shuffle=True,
                        random_state=RANDOM_STATE,
                        test_size=self.params["holdout"]
                    )

            elif self.params["task_type"] == "regression": 
                self.state["SX"], self.state["TX"] = \
                    train_test_split(
                        df,
                        shuffle=True,
                        random_state=RANDOM_STATE,
                        test_size=self.params["holdout"]
                    )

                self.state["X"] = self.state["SX"][feature_columns]
                self.state["y"] = self.state["SX"][label_column]

                self.state["HX"] = self.state["TX"][feature_columns]
                self.state["Hy"] = self.state["TX"][label_column]
            
                if self.params["task_type"] == "regression":
                    self.state["Ay"] = self.state["Ay"].apply(pd.to_numeric)
                    self.state["Hy"] = self.state["Hy"].apply(pd.to_numeric)
                    self.state["y"]  = self.state["y"].apply(pd.to_numeric)

        else: 
            self.state["X"] = self.state["AX"]
            self.state["y"] = self.state["Ay"]

            if self.params["task_type"] == "regression":
                self.state["Ay"] = self.state["Ay"].apply(pd.to_numeric)
                self.state["y"]  = self.state["y"].apply(pd.to_numeric)

        # remove label from grouped types 
        grouped_types = self.state["dataset"].state["grouped_types"]
        label_type = self.state["dataset"].state["types"][label_column]
        grouped_types[label_type].remove(label_column)

        
# =========================================================================== #