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

# =========================================================================== #

# ==== PROJECT IMPORTS ====================================================== #
from core.helpers.classifiers import *
from core.helpers.regressors import *
from core.helpers.common_imports import * 
from core.helpers.project_imports import * 


from core.helpers.presets import *
# =========================================================================== #

# ==== MAIN SCRIPT ========================================================== # 

class Splitter: 
    def __init__(
        self, 
        dataset = load_wine_dataset(),
        kfold_params = {
            "n_splits"     : 2, 
            "n_repeats"    : 5,
            "random_state"  : RANDOM_STATE
        },
        kfold_type = "stratified",
        verbose = True,
        indent = ""
    ) :    
        """ 
            Acts as a wrapper to Scikit-Learn's RepeatedStratifiedKFold and 
            RepeatedKFold for splitting arbitary datasets.

            Parameters: 
                labeled_dataset      
                    - the labeled dataset to assigned to this object 

                kfold_params
                    - the params to pass on kfold

                kfold_type = "stratified" 
                    - whether "stratified" or "normal"
        """ 

        # messages #
        self.verbose = verbose 
        self.indent = indent 

        # main parameters # 
        self.params = {
            "dataset" : dataset,
            "kfold_params" : kfold_params, 
            "kfold_type" : kfold_type,
            "verbose" : verbose, 
            "indent" : indent
        } 
        
        # state objects # 
        self.state = {
            "dataset" : dataset,
            "kfold" : None,
            "indices" : None
        }

        # flow
        self.make_kfold_object()
        self.determine_indices()
   
    ###########################################################################

    def make_kfold_object(self): 
        params = {
            **{ 
                "n_repeats" : 5, 
                "n_splits" : 2, 
                "random_state" : RANDOM_STATE 
            },
            **self.params["kfold_params"]
        }

        if self.params["kfold_type"] == "normal": 
            self.state["kfold"] = RepeatedKFold(**params) 
        elif self.params["kfold_type"] == "stratified": 
            self.state["kfold"] = RepeatedStratifiedKFold(**params)
        else: 
            raise Exception(f"Unknown kfold type {self.state['kfold']['type']}")
        
    
    
    ###########################################################################

    def determine_indices(self): 
        kfold = self.state["kfold"]

        X = self.state["dataset"].state["X"]
        y = self.state["dataset"].state["y"]
        
        if type(kfold) is RepeatedKFold:
            self.state["indices"] = list(kfold.split(X)) 
        elif type(kfold) is RepeatedStratifiedKFold:
            self.state["indices"] = list(kfold.split(X, y)) 
        else:
            raise Exception(f"Unknown kfold object [{kfold}]")
    
    ###########################################################################

    def train_split(self, fold_no): 
        train_indices = self.train_indices(fold_no)
        X = self.state["dataset"].state["X"].iloc[train_indices]
        y = self.state["dataset"].state["y"].iloc[train_indices]
        return X, y

    ###########################################################################

    def test_split(self, fold_no): 
        test_indices = self.test_indices(fold_no)
        X = self.state["dataset"].state["X"].iloc[test_indices]
        y = self.state["dataset"].state["y"].iloc[test_indices]
        return X, y

    ###########################################################################

    def train_indices(self, fold_no):
        return self.state["indices"][fold_no][0]

    ###########################################################################

    def test_indices(self, fold_no): 
        return self.state["indices"][fold_no][1]

    ###########################################################################

    def train_size(self, fold_no): 
        return len(self.state["indices"][fold_no][0]) 

    ###########################################################################

    def test_size(self, fold_no):
        return len(self.state["indices"][fold_no][1]) 
   
    ###########################################################################

    def train_sizes(self): 
        return [len(indices[0]) for indices in self.state["indices"]]
    
    ###########################################################################

    def test_sizes(self):
        return [len(indices[1]) for indices in self.state["indices"]]

    ###########################################################################
    
    def fold_count(self):
        return len(self.state["indices"])

# =========================================================================== #