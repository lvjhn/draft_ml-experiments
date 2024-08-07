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

# --- general parameters --- #
context          = 2024
geo_level        = "provinces"
name_field       = "Province"

# --- path to files and folder --- #
metadata_file    = \
    f"./data/locations/{context}/metadata/{geo_level}.csv"
articles_folder  = \
    f"./data/locations/{context}/wiki-articles/preprocessed/{geo_level}/"

# --- load metadata file to get list of items --- # 
df = pd.read_csv(metadata_file)
names = df[name_field]
links = df["Article Link_x"]

# --- loop through names and collect paragraphs and documents ---  # 
documents  = { "text" : [], "label" : [] } 
paragraphs = { "text" : [], "label" : [] }

i = 0
n = len(names)
for index, item in df.iterrows(): 
    name = item[name_field]
    print(f"@ Processing {name} ({i + 1}/{n})")
    link = item["Article Link_x"]
    suffix = link.split("/")[-1]
    ifile = f"{articles_folder}{suffix}.json"
    content = json.load(open(ifile, "r"))
    pars = content["data"] 

    # add document #
    document = "\n".join([".".join(paragraph) for paragraph in pars])
    documents["text"].append(document)
    documents["label"].append(name)

    # add paragraphs #
    for par in pars:
        par_ = " ".join(par) 
        paragraphs["text"].append(par_)
        paragraphs["label"].append(name)
        
    i += 1

paragraphs_df = pd.DataFrame(paragraphs) 
documents_df  = pd.DataFrame(documents)

print("@ Saving files.")

os.makedirs(
    f"./data/datasets/wikipedia/{geo_level}/documents",
    exist_ok=True
)

os.makedirs(
    f"./data/datasets/wikipedia/{geo_level}/paragraphs",
    exist_ok=True
)


documents_df.to_csv(
    f"./data/datasets/wikipedia/{geo_level}/documents/all.csv"
)
paragraphs_df.to_csv(
    f"./data/datasets/wikipedia/{geo_level}/paragraphs/all.csv"
)

print("@ Done.")