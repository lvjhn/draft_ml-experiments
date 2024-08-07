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

class TrainTestSplitExperiment:
    def __init__(
        self, 
        dataset = None,
        tts = None, 
        pipeline = None,
        scoring = None,
        verbose = True, 
        indent = False
    ):  
        # logging 
        self.verbose = verbose 
        self.indent  = indent 
        
        # parameters 
        self.params = {
            "dataset" : dataset,
            "tts" : tts,
            "scoring" : scoring,
            "pipeline" : pipeline
        }

        # state 
        self.state = {
            "dataset" : dataset
        }

    def dataset_name(self): 
        dataset = self.state["dataset"].state["dataset"]
        context = dataset.params["context"]
        subcontext = dataset.params["subcontext"]
        name = dataset.params["name"] 
        file = dataset.params["file"]
        return f"{context}:{subcontext}/{name}:{file}"
       
    def features_and_label(self):
        features = self.state["dataset"].state["feature_columns"]
        label = self.state["dataset"].state["label_column"]
        return features, label

    def label_counts(self, y):
        return Common.get_value_counts(y)

    def evaluate(self):
        # print info. 
        features, label = self.features_and_label()
        pipeline = self.params["pipeline"](None, None, None, None)

        print(f"> Evaluating [{self.dataset_name()}]")
        print(f"\t:: Feature Columns   : {features}")
        print(f"\t:: Label Column      : {label}")
        print(f"\t:: Pipeline          : {type(pipeline)}")

        # get train and test split
        X_train, X_test, y_train, y_test = self.params["tts"]
        labels = list(Common.get_value_counts(y_train).keys())
        
        X_holdout = self.state["dataset"].state["HX"]
        y_holdout = self.state["dataset"].state["Hy"]

        print(f"> Evaluation Info")
        print(f"\t:: Train Size         : {len(X_train)}")        
        print(f"\t:: Test Size          : {len(X_test)}")

        # get label counts
        print(f"> Label Counts") 
        print("\t:: Train")
        Common.print_dict(self.label_counts(y_train), "\t\t")
        print("\t:: Test")
        Common.print_dict(self.label_counts(y_test), "\t\t")

        # run experiment
        print("> Training...")
        model = self.params["pipeline"](X_train, y_train, X_test, y_test) 
        model.pipeline.fit(X_train, y_train)
        print("\t:: Done.")

        # evaluation
        print("> Generating predictions...")

        print("\t:: Predicting on train set.")
        y_pred_train = model.pipeline.predict(X_train)
        print()
        print()

        print("\t:: Predicting on test set.")
        y_pred_test = model.pipeline.predict(X_test)    
        print()
        print()
        
        print("\t:: Predicting on holdout set.")
        y_pred_holdout = model.pipeline.predict(X_holdout)
        print()
        print()

        # evaluation
        print("> Evaluating predictions...")

        print(":: Confusion Matrix (Train Set).")
        matrix = confusion_matrix(y_train, y_pred_train)
        Common.print_conf_matrix(labels, matrix)
        print()

        print(":: Confusion Matrix (Test Set)")
        matrix = confusion_matrix(y_test, y_pred_test)
        Common.print_conf_matrix(labels, matrix)
        print()


        print(":: Confusion Matrix (Holdout Set)")
        matrix = confusion_matrix(y_holdout, y_pred_holdout)
        Common.print_conf_matrix(labels, matrix)
        print()


        print("> Computing scores...")

        print(":: Accuracy")
        print(f"\ttrain   = {accuracy_score(y_train, y_pred_train)}")
        print(f"\ttest    = {accuracy_score(y_test, y_pred_test)}")
        print(f"\tholdout = {accuracy_score(y_holdout, y_pred_holdout)}")
        print()

        print(":: Balanced Accuracy")
        print(f"\ttrain   = {balanced_accuracy_score(y_train, y_pred_train)}")
        print(f"\ttest    = {balanced_accuracy_score(y_test, y_pred_test)}")
        print(f"\tholdout = {balanced_accuracy_score(y_holdout, y_pred_holdout)}")
        print()

        print(":: Input Size")
        print(f"\tinput_size = {model.input_size}")
        print()

