# =========================================================================== #
# :: [NAME]
#
# :: Description ::
#
#
# =========================================================================== #

# ==== COMMON IMPORTS ======================================================= #
from setup.settings import *
from core.helpers.major_imports import *
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
# =========================================================================== #

# ==== MAIN SCRIPT ========================================================== # 

def load_agnc_dataset(
    label = "Class Index", 
    holdout = 0.3,
    task_type = "classification",
    features = ["Title", "Description"]
):
    dataset = Dataset(
        context = "kaggle",
        subcontext = "amananandrai",
        name = "AG News Classification Dataset",
        file = "all.csv",
        manual_types = {}
    )


    labeled = LabeledDataset(
        dataset      = dataset,
        feature_mode = "include",
        features     = features,
        label        = label,
        remove_ids   = True,
        holdout      = holdout,
        task_type    = task_type
    )

    return labeled 

# ============================================================================ #

def load_nac_b_dataset(
    label = "category", 
    holdout = 0.3,
    task_type = "classification",
    features =  ["headlines", "description", "content"]
):
    dataset = Dataset(
        context = "kaggle",
        subcontext = "banuprakashv",
        name = "News Article Classification Dataset",
        file = "all.csv",
        manual_types = {}
    )


    labeled = LabeledDataset(
        dataset      = dataset,
        feature_mode = "include",
        features     = features,
        label        = label,
        remove_ids   = True,
        holdout      = holdout,
        task_type    = task_type
    )

    return labeled 

# ============================================================================ #

def load_nt_dataset(
    label = "category", 
    holdout = 0.3,
    task_type = "classification",
    features = ["text"]
):
    dataset = Dataset(
        context = "kaggle",
        subcontext = "dabayondharchowdhury",
        name = "News Text",
        file = "all.csv",
        manual_types = {}
    )


    labeled = LabeledDataset(
        dataset      = dataset,
        feature_mode = "include",
        features     = features,
        label        = label,
        remove_ids   = True,
        holdout      = holdout,
        task_type    = task_type
    )

    return labeled 

# ============================================================================ #

def load_wnc_dataset(
    label = "label", 
    holdout = 0.3,
    task_type = "classification",
    features = ["text"]

):
    dataset = Dataset(
        context = "kaggle",
        subcontext = "khoshbayani",
        name = "World News Category",
        file = "all.csv",
        manual_types = {}
    )


    labeled = LabeledDataset(
        dataset      = dataset,
        feature_mode = "include",
        features     = features,
        label        = label,
        remove_ids   = True,
        holdout      = holdout,
        task_type    = task_type
    )

    return labeled

# ============================================================================ #

def load_ncd_r_dataset(
    label = "category", 
    holdout = 0.3,
    task_type = "classification",
    features = ["headline", "short_description"]
):
    dataset = Dataset(
        context = "kaggle",
        subcontext = "rmisra",
        name = "News Category Dataset",
        file = "all.csv",
        manual_types = {}
    )


    labeled = LabeledDataset(
        dataset      = dataset,
        feature_mode = "include",
        features     = features,
        label        = label,
        remove_ids   = True,
        holdout      = holdout,
        task_type    = task_type
    )

    return labeled

# ============================================================================ #

def load_nac_t_dataset(
    label = "category", 
    holdout = 0.3,
    task_type = "classification",
    features = ["title", "body"]
):
    dataset = Dataset(
        context = "kaggle",
        subcontext = "timilsinabimal",
        name = "News Article Category",
        file = "all.csv",
        manual_types = {}
    )


    labeled = LabeledDataset(
        dataset      = dataset,
        feature_mode = "include",
        features     = features,
        label        = label,
        remove_ids   = True,
        holdout      = holdout,
        task_type    = task_type
    )

    return labeled

# ============================================================================ #

def load_pfncd_dataset(
    label = "Label", 
    holdout = 0.3,
    task_type = "classification",
    features = ["Headline", "Content"]
):
    dataset = Dataset(
        context = "github",
        subcontext = "aaroncarlfernandex",
        name = "Philippine Fake News Corpus",
        file = "all.csv",
        manual_types = {}
    )


    labeled = LabeledDataset(
        dataset      = dataset,
        feature_mode = "include",
        features     = features,
        label        = label,
        remove_ids   = True,
        holdout      = holdout,
        task_type    = task_type
    )

    return labeled
