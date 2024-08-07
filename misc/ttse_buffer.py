#
# Define a straightforward train-test split experiment.
#
 
from core.helpers.project_imports import *
from setup.settings import *
from core.experiment.ttse import TrainTestSplitExperiment
from loaders.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

# --- SET MAJOR PARAMETER ---------------------------------------------------- #
dataset = load_pnfc_dataset()
# ---------------------------------------------------------------------------- #

def define_scoring(y_true, y_pred): 
    return {
        "accuracy" : accuracy_score(y_true, y_pred),
        "balanced_accuracy" : balanced_accuracy_score(y_true, y_pred)
    }

def define_pipeline(X_train, y_train, X_test, y_test):
    pipeline = BL_C_MLPC_10()

    pipeline.create_pipeline(dataset, "classification")

    pipeline.dataset = dataset

    pipeline.X_train = X_train
    pipeline.y_train = y_train 
    pipeline.X_test  = X_test 
    pipeline.y_test  = y_test

    return pipeline

ttse = TrainTestSplitExperiment(
    dataset = dataset,
    tts = train_test_split(
        dataset.state["X"], 
        dataset.state["y"], 
        stratify=dataset.state["y"],
        test_size=0.3, 
        random_state=RANDOM_STATE
    ), 
    pipeline = define_pipeline
) 

ttse.evaluate()