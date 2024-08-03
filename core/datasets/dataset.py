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
# =========================================================================== #

# ==== MAIN SCRIPT ========================================================== # 

class Dataset: 
    def __init__(
        self, 
        
        context = "kaggle",
        subcontext = "uciml",
        name = "Wine",
        file = "wine.data",

        manual_types = {},

        impute_missing = {
            "cbo"      : "most_frequent",
            "numeric"  : "mean",
            "others"   : "most_frequent"
        },

        verbose = True, 
        indent = True, 
        load = True,

        DATASETS_PATH = DATASETS_PATH
    ):    
        """ 
            Loads a dataset from file and wraps it as an object.
                
                Parameters:
                    context   
                        - context folder where to look for the dataset 
                          e.g. kaggle, uci, data.world 

                    subcontext  
                        - user or organization under the context who
                          distributed the dataset 

                    name        
                        - name of the dataset 
                    
                    file        
                        - filename of the dataset   

                    verbose     
                        - whether to display log messages

                    impute_missing 
                        - impute missing variables for a given type

                    indent      
                        - indentation level for messages

                    load 
                        - whether to load the dataset automatically or not

                    DATASET_PATH 
                        - where to load dataset from 
        """ 

        # messages #
        self.verbose = verbose 
        self.indent = indent 

        # main parameters # 
        self.params = {
            "context" : context ,
            "subcontext" : subcontext, 
            "name" : name, 
            "file" : file,
            "manual_types" : manual_types,
            "impute_missing" : impute_missing,
            "verbose" : verbose, 
            "indent" : indent,
            "load" : load,
            "DATASETS_PATH" : DATASETS_PATH
        } 
        
        # state objects # 
        self.state = {
            "df" : None,
            "types" : {}
        }

        # localized objects
        self.verbose = verbose 
        self.indent = indent

        # load the dataset 
        if load:
            self.load()

    ###########################################################################

    def context_path(self): 
        DATASETS_PATH = self.params["DATASETS_PATH"]
        context = self.params["context"]
        return f"./{DATASETS_PATH}/{context}/"

    ###########################################################################

    def subcontext_path(self):
        DATASETS_PATH = self.params["DATASETS_PATH"]
        context = self.params["context"]
        subcontext = self.params["subcontext"]
        return f"./{DATASETS_PATH}/{context}/{subcontext}/"

    ###########################################################################
    
    def folder_path(self):
        DATASETS_PATH = self.params["DATASETS_PATH"]
        context = self.params["context"]
        subcontext = self.params["subcontext"]
        name = self.params["name"]
        return f"./{DATASETS_PATH}/{context}/{subcontext}/{name}/"

    ###########################################################################

    def file_path(self): 
        DATASETS_PATH = self.params["DATASETS_PATH"]
        context = self.params["context"]
        subcontext = self.params["subcontext"]
        name = self.params["name"]
        file = self.params["file"]
        return f"{DATASETS_PATH}/{context}/{subcontext}/{name}/{file}"

    ###########################################################################

    def load(self): 
        self.state["df"] = pd.read_csv(self.file_path())
        self.refresh()
        self.state["df"] = \
            self.state["df"].sample(
                len(self.state["df"]),
                random_state=RANDOM_STATE
            )
        return self.state["df"]

    ###########################################################################

    def refresh(self):
        self.detect_types()
        self.group_types()

    ###########################################################################

    def columns(self): 
        return self.state["df"].columns 

    ###########################################################################

    def df(self): 
        return self.state["df"]

    ###########################################################################

    def detect_types(self):
        columns = self.columns()
        
        # automatically detect data types
        df = self.df()
        
        types = {}
        for column in columns: 
            type_, values = Common.detect_type(column, df[column])
            types[column] = type_
            df[column] = values

        # apply manual typing 
        manual_types = self.params["manual_types"]
        for column in manual_types:
            type_ = manual_types[column]
            types[column] = type_

        self.state["types"] = types
    
    ###########################################################################

    def group_types(self): 
        self.state["grouped_types"] = defaultdict(set)
        grouped_types = self.state["grouped_types"]

        types_ = self.state["types"]
        for column in types_:
            type_ = types_[column]
            grouped_types[type_].add(column)

    ###########################################################################
        

# =========================================================================== #