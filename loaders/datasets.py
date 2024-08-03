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

def weather_classification_dataset(
    label = "Weather Type", 
    holdout = 0.3,
    task_type = "classification"
):
    dataset = Dataset(
        context = "kaggle",
        subcontext = "nikhil7280",
        name = "Weather Type Classification",
        file = "weather_classification_data.csv",
        manual_types = {}
    )


    labeled = LabeledDataset(
        dataset      = dataset,
        feature_mode = "include",
        features     = None,
        label        = label,
        remove_ids   = True,
        holdout      = holdout,
        task_type    = task_type
    )

    return labeled 

# ============================================================================ #

def website_classification_dataset(
    label = "Category", 
    holdout = 0.3,
    task_type = "classification"
):
    dataset = Dataset(
        context = "kaggle",
        subcontext = "hetulmehta",
        name = "Website Classification",
        file = "website_classification.csv", 
        manual_types = {
            "cleaned_website_text" : "text"
        }
    )

    labeled = LabeledDataset(
        dataset      = dataset,
        feature_mode = "include",
        features     = ["cleaned_website_text"],
        label        = label,
        remove_ids   = True,
        holdout      = holdout,
        task_type    = task_type
    )

    return labeled 

# ============================================================================ #

def imdb_movie_reviews_dataset(
    label = "label", 
    holdout = 0.3,
    task_type = "classification"
):
    dataset = Dataset(
        context = "kaggle",
        subcontext = "lakshmi25npathi",
        name = "IMDB Dataset",
        file = "IMDB Dataset.csv",
        manual_types = {
            "review" : "text",
        }
    )

    labeled = LabeledDataset(
        dataset      = dataset,
        feature_mode = "include",
        features     = None,
        label        = "sentiment",
        remove_ids   = True,
        holdout      = holdout,
        task_type = task_type
    )
    
    return labeled 

# ============================================================================ #

def ag_news_classification_dataset(
    label = "label", 
    holdout = 0.3,
    task_type = "classification"
):
    dataset = Dataset(
        context = "kaggle",
        subcontext = "amananandrai",
        name = "AG News Classification Dataset",
        file = "ag-news.csv",
        manual_types = {
            "text" : "text",
        }
    )
    
    labeled = LabeledDataset(
        dataset      = dataset,
        feature_mode = "include",
        features     = None,
        label        = "label",
        remove_ids   = True,
        holdout      = holdout,
        task_type = task_type
    )
    return labeled 

# ============================================================================ #

def philippine_fake_news_corpus_dataset(
    label = "Label", 
    holdout = 0.3,
       task_type = "classification"
):
    dataset = Dataset(
        context = "github",
        subcontext = "aaroncarlfernandex",
        name = "Philippine Fake News Corpus",
        file = "Philippine Fake News Corpus.csv",
        manual_types = {
            "Headline" : "text",
            "Content" : "text"
        }
    )

    labeled = LabeledDataset(
        dataset      = dataset,
        feature_mode = "include",
        features     = ["Content"],
        label        = label,
        remove_ids   = True,
        holdout      = holdout,
        task_type = task_type
    )
    
    return labeled 


