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
from core.helpers.common_imports import *
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

from loaders.datasets import *
# =========================================================================== #

models = []
cache_dict = {}

# provinces_df = pd.read_csv("./data/locations/2024/metadata/provinces.csv")
# provinces = list(provinces_df["Province"])

provinces = [
    "Ilocos Sur",
    "Ilocos Norte",
    "Metro Manila",
    "Albay",
    "Catanduanes",
    "Sorsogon",
    "Masbate",
    "Camarines Norte",
    "Camarines Sur",
    "Davao del Norte",
    "Davao del Sur",
    "Cebu"
]

i = 0 
n = len(provinces)
for province in provinces:
    print(f"@ Training for {province} ({i + 1}/{n}).")

    # load and prepare dataset
    print("\t> Loading dataset.")
    dataset = Dataset(
        context = "wikipedia",
        subcontext = "provinces",
        name = "documents", 
        file = "all.csv",
        manual_types = {
            "text" : "text"
        }
    )

    def relabel(x):
        if x == province:
            return province 
        else: 
            return f"(Not) {province}"

    # relabel
    print(f"\t> Relabeling.")
    dataset.state["df"]["label"] = \
        dataset.state["df"]["label"].apply(relabel)
    dataset.refresh()

    # set up labeled dataset
    print(f"\t> Setting up labeled dataset.")
    labeled = LabeledDataset(
        dataset = dataset, 
        label = "label", 
        features = ["text"],
        holdout=None
    )

    # prepare and train model 
    print(f"\t> Training model.")
    model = BL_C_MLPC_10()
    model.create_pipeline(labeled, "classification")
    model.pipeline.fit(labeled.state["X"], labeled.state["y"])

    # add model 
    print(f"\t> Adding model.")
    models.append(model.pipeline)
    
    i += 1

print("@ Saving model.")
pickle.dump(models, open("model.pickle", "wb"))



