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

class Sampler: 
    def __init__(
        self, 
        dataset = load_wine_dataset(),
        kfold_params = {
            "n_splits"     : 2, 
            "n_repeats"    : 5,
            "random_state"  : RANDOM_STATE
        },
        kfold_type = "stratified",
        model_evaluation_enabled = True,
        n_sample_sizes = 100,
        sampler_min_bound = 50,
        task_type = "classification",
        verbose = True,
        indent = ""
    ) :    
        """ 
            Samples a dataset for 50 different training sizes from 
            10% of the dataset up to 100% of the dataset.

            Parameters: 
                dataset      
                    - the labeled dataset to assigned to this object 

                kfold_params
                    - the params to pass on kfold

                n_sample_sizes
                    - no. of different sample sizes from 10% of the data
                      to 100% of the data

                model_evaluation_enabled 
                    - whether model evaluation, fitting, predicting, and scoring
                      will be applied
                    - if False, the learning curve can be used to report the 
                      dataset across different sample sizes 
                    - if True, the learning curve can be used to report the 
                      model's performance across different sample size

                verbose 
                    - turn log messages on or off 

                indent 
                    - indentation level for messages
        """ 

        # messages #
        self.verbose = verbose 
        self.indent = indent 

        # main parameters # 
        self.params = {
            "dataset" : dataset,
            "kfold_params" : kfold_params, 
            "kfold_type" : kfold_type,
            "n_sample_sizes" : n_sample_sizes,
            "model_evaluation_enabled" : model_evaluation_enabled,
            "sampler_min_bound" : sampler_min_bound,
            "task_type" : task_type,
            "verbose" : verbose,
            "indent" : indent
        } 
        
        # state objects # 
        self.state = {
            "dataset"       : dataset,
            "total_sizes"   : [],
            "train_sizes"   : [],
            "test_sizes"    : [],
            "kfolds"        : []
        }

        # flow
        self.make_sizes()
    
    ###########################################################################

    def make_splitter_for_sample_size(self, i, n, sample_size):
        self.verbose and \
            print(f"{self.indent}==> Making random splitter ({i + 1} of {n})")
                

        dataset = self.state["dataset"]
        feature_columns = dataset.state["feature_columns"]
        label_column = dataset.state["label_column"]
        
        # concatenate X and y 
        X = self.state["dataset"].state["X"]
        y = pd.DataFrame(self.state["dataset"].state["y"])

        frac = sample_size / X.shape[0]

        Xy = pd.merge(X, y, left_index=True, right_index=True)
        
        print("kfold_type : ", self.params["kfold_type"] )
        
        if self.params["kfold_type"] == "stratified":
            Xy = Xy.groupby(label_column, group_keys=False,)\
                .apply(lambda x: x.sample(frac=frac, random_state=RANDOM_STATE))
        else: 
            Xy = Xy.sample(frac=frac, random_state=RANDOM_STATE)
       
        # create dataset object for this sample
        dataset = Dataset(
            context = "Learning Curves",
            subcontext = "Split",
            name = sample_size, 
            file = None,
            load = False,
            manual_types = \
                dataset.state["dataset"].params["manual_types"]
        )
        
        dataset.state["df"] = Xy 
        dataset.refresh()

        task_type = self.state["dataset"].params["task_type"]

        # create labeled dataset for this dataset object
        labeled_dataset = LabeledDataset(
            dataset = dataset, 
            features = feature_columns, 
            label = label_column,
            task_type = task_type,
            holdout = None
        )

        # create splitter for this dataset object 
        splitter = Splitter(
            dataset = labeled_dataset,
            kfold_params = self.params["kfold_params"],
            kfold_type = self.params["kfold_type"]
        )

        self.state["kfolds"].append(splitter)
        
        # get train sizes 
        train_size = int(np.mean(splitter.train_sizes()))
        test_size  = int(np.mean(splitter.test_sizes()))

        self.state["train_sizes"].append(sample_size - test_size)
        self.state["test_sizes"].append(test_size)
        self.state["total_sizes"].append(sample_size)

        self.verbose and \
            print(f"\tSize (Train)  : {train_size}")
        self.verbose and \
            print(f"\tSize (Test)   : {test_size}")
        self.verbose and \
            print(f"\tSize (Total)  : {train_size + test_size}")


    ###########################################################################
    
    def make_sizes(self): 
        n_sample_sizes = self.params["n_sample_sizes"]
        n_data_points  = self.state["dataset"].state["X"].shape[0]
        
        lb = self.params["sampler_min_bound"]
        
        if n_sample_sizes == 1:
            lb = n_data_points * 1.0 
            
        ub = n_data_points * 1.0
        
        sample_space = np.linspace(lb, ub, n_sample_sizes) 

        self.total_sizes = list([int(x) for x in sample_space])
        
        i = 0
        n = len(self.total_sizes)
        for size in self.total_sizes:
            self.make_splitter_for_sample_size(i, n, size)
            i += 1

    ###########################################################################
    