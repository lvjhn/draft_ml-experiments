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
from bottle import route, run, template
# =========================================================================== #

# ==== PROJECT IMPORTS ====================================================== #
from core.helpers.classifiers import *
from core.helpers.regressors import *
from core.helpers.common_imports import * 
from core.helpers.project_imports import * 
# =========================================================================== #


def get_model_name(name):
    if name == "fasttext": 
        return "fasttext-wiki-news-subwords-300"
    elif name == "glove":
        return "glove-wiki-gigaword-300"
    elif name == "word2vec":
        return "word2vec-google-news-300"
    else:
        print(f"Unknown model name {name}.")
        return

# ==== MAIN SCRIPT ========================================================== # 

# --- Load Model ------------------------------------------------------------ #
model_name = sys.argv[1]

print(f"@ Loading model {model_name}.")
model = AGMV(model_name=model_name, dims=int(model_name.split("-")[-1]))
print(f"@ Loaded model.")

# --------------------------------------------------------------------------- #
def get_vector(sentence):
    return model.fit_transform(pd.DataFrame([sentence]))[0]

# ------------------------------------------------------------------------ #

print("=====================================================================")

@route('/transform/<sentence>')
def transform(sentence):
    vector = get_vector(sentence)
    print(vector)
    return json.dumps(vector, default=Common.format_json, indent=4)

@route('/cache/<context>/<subcontext>/<name>/<columns>')
def transform(context, subcontext, name, columns):
    try:
        print()
        
        print(f"@ Loading dataset [{context} / {subcontext} / {name}].")
        dataset = Dataset(
            context = context, 
            subcontext = subcontext,
            name = name,
            file = "all.csv"
        )
        print(f"\t> Loaded dataset [{len(dataset.state['df'])} rows].")
        
        print()

        print("@ Loading dataset.")
        columns = columns.split("|")
        print(f"@ Caching for columns: {columns}")

        folder = \
            f"./data/cache/vectors/{model_name}/{context}/{subcontext}/{name}/"

        for i in range(len(columns)): 
            column = columns[i]
            print(f"\t> Processing {column} ({i + 1}/{len(columns)})")
            for averaging in ["word", "sentence"]:
                out_folder = f"{folder}{averaging}/{column}"

                os.makedirs(out_folder, exist_ok=True)

                print(f"\t\t:: In {averaging}-level averaging.")

                df = dataset.state["df"]
                df = df.replace(np.nan, '', regex=True)

                texts   = pd.Series(df[column]) 
                index   = np.array(texts.index)
                vectors = model.fit_transform(texts)

                print(f"\t\t\t:: Saving index to index.npy")
                np.save(out_folder + "/index.npy", index)

                print(f"\t\t\t:: Saving vectors to vectors.npy")
                np.save(out_folder + "/vectors.npy", vectors)

        print()

    except Exception as e:
        print(repr(e))

    return

run(host='localhost', port=8080)

# =========================================================================== #