import sys 
import numpy as np
from core.helpers.major_imports import *

from gensim import downloader as api

from setup.transforms.agmv import AGMV
from core.datasets.dataset import Dataset

# get target from CLI arguments 
argv   = sys.argv

target   = argv[1].split("|") 
columns  = argv[2].split("|")

# --- get target tokens --- #
context    = target[0]
subcontext = target[1]
name       = target[2] 
file       = target[3] 

# --- models --- #
models     = [
    # --- FASTTEXT --- #
    ("fasttext-wiki-news-subwords-300", 300),

    # --- GLOVE --- #
    ("glove-wiki-gigaword-300", 300),

    # --- WORD2VEC --- #
    ("word2vec-google-news-300", 300)
]

# --- load dataset --- # 
dataset = Dataset(
    context = context,
    subcontext = subcontext,
    name = name, 
    file = file
)

# ---- precompute vectors for each column --- # 
def cache_for_column(dataset, column): 
    for model in models:
        cache_for_model(dataset, column, model)

# ---- precompute vectors for each model --- # 
def cache_for_model(dataset, column, model):
    print(f"> Caching {column} for {model[0]}.")

    # create output folder
    output_folder = (
        f"./data/cache/vectors/" + 
        f"/{context}/{subcontext}/{name}/{column}/" + 
        f"/{model[0]}/"
    )

    os.makedirs(output_folder, exist_ok=True)

    # for each averaging type (word and sentence) 
    os.makedirs(output_folder + "/word", exist_ok=True)
    os.makedirs(output_folder + "/sentence", exist_ok=True)

    for averaging in ["word", "sentence"]:
        folder = output_folder + "/" + averaging + "/"

        # transform columns
        X     = pd.Series(dataset.state["df"][column])
        index = X.index
        
        # save index 
        np.save(folder + "index.npy", index)        

        # load model
        model_ = AGMV(
            model_name = model[0], 
            averaging = averaging,
            dims = model[1]
        ) 

        # transform data
        X = model_.fit_transform(X) 
        
        # save data to folder
        np.save(f"{folder}/vectors.npy", X)

    del AGMV.CACHE[model[0]]

    print()


# ---- cache --- # 
for column in columns:
    cache_for_column(dataset, column)