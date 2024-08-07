# =========================================================================== #
# :: AGMV
#
# :: Description ::
#   Averaged Gensim Model Vectorizer
#   - averages the vectors produced by gensim models for each word 
#     to represent a document-
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
from sklearn.base import BaseEstimator, TransformerMixin
from gensim import downloader as api
from gensim.utils import simple_preprocess
# =========================================================================== #

# ==== PROJECT IMPORTS ====================================================== #
from core.helpers.classifiers import *
from core.helpers.regressors import *
from core.helpers.common_imports import * 
from core.helpers.project_imports import * 

from setup.pipelines.extra_transforms import *
from gensim import downloader as api
# =========================================================================== #

MODEL_CACHE  = {}
VECTOR_CACHE = {}
WORD_CACHE   = {}

class AGMV(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        model_name = "glove-twitter-25",
        model_obj = None,
        use_server = False,
        pipeline = None,
        dims = 25,
        averaging = "word",
        use_cache = False,
        cache_dict = None,
        verbose = True,
        column = None
    ): 
        # verbose flag
        self.verbose = verbose

        # identify model name
        self.model_name = model_name 

        # identify model name
        self.model_obj = model_obj 

        # identify dimensions 
        self.dims = dims

        # identify averaging level 
        self.averaging = averaging

        # whether to use cache or not 
        self.use_cache = use_cache

        # cache dictionary
        self.cache_dict = cache_dict

        # whether to use server or not 
        self.use_server = use_server

        # the pipeline attached to this transfomer
        self.pipeline = pipeline

        # column name 
        self.column = column

        # load model if not yet in cache
        if self.use_cache == False:
            if model_name not in MODEL_CACHE: 
                if self.verbose:
                    print(f"===> AGMV : Loading model `{self.model_name}` <===")
                if model_obj is not None: 
                    MODEL_CACHE[model_name] = model_obj 
                else:
                    MODEL_CACHE[model_name] = api.load(model_name) 

            self.model = MODEL_CACHE[model_name]


    # ======================================================================== #

    def get_dataset_info(self):
        dataset = self.pipeline.dataset 
        base_dataset = dataset.state["dataset"] 
        context = base_dataset.params["context"]
        subcontext = base_dataset.params["subcontext"] 
        name = base_dataset.params["name"]
        file = base_dataset.params["file"] 
        model_name = self.model_name
        averaging = self.averaging
        return context, subcontext, name

    # ======================================================================== #
    
    def load_cache(self): 
        averaging = self.averaging
        model_name = self.model_name
        context, subcontext, name = self.get_dataset_info()
        column = self.column

        # load cache
        if self.verbose: 
            print(f"=== Loading cache for {column}  ===")
            
      
        cache_folder = (
            f"./data/cache/vectors/{model_name}" + \
            f"/{context}/{subcontext}/{name}" + \
            f"/{averaging}"
        )
        
        if self.verbose:
            print(f"=== Cache Folder : {cache_folder}  ===")


        index_file   = f"{cache_folder}/{column}/index.npy"
        vectors_file = f"{cache_folder}/{column}/vectors.npy"

        # build cache
        if self.verbose: 
            print("=== Building cache. ===")

        index = np.load(index_file)
        vectors = np.load(vectors_file)

        cache = {} 

        i = 0
        for index_ in index:
            print(f"=== Loading cache item {i + 1} / {len(index)} ===", end="\r")
            cache[index_] = vectors[i, :]
            i += 1

        VECTOR_CACHE[f"{context}|{subcontext}|{name}|{column}"] = cache

    # ======================================================================== #

    def vectorize_word(self, w):
        if w not in self.model:
            return self.model["unk"]
        return self.model[w]

    # ======================================================================== #

    def vectorize_sentence(self, s):
        average = np.zeros(self.dims)
        valid_words = 0
        for i in range(len(s)):
            w = s[i]
            v = self.vectorize_word(w)
            average += v
            valid_words += 1
        average = average / valid_words
        return average

    # ======================================================================== #

    def fit(self, X, y = None): 
        return self

    # ======================================================================== #

    def preprocess_text(self, a, index): 
        if not self.use_cache and not self.use_server:
            from nltk import tokenize

            vector           = None
            
            if self.cache_dict is not None: 
                if index in self.cache_dict: 
                    return self.cache_dict[index]

            if self.averaging == "word":
                words       = tokenize.word_tokenize(a) 
                vector      = [self.vectorize_word(w) for w in words]

            elif self.averaging == "sentence":
                sentences  = \
                    tokenize.sent_tokenize(a) 
                sentences  = \
                    [tokenize.word_tokenize(s) for s in sentences]
                vector     = \
                    [self.vectorize_sentence(s) for s in sentences]

            vector = np.mean(vector, axis=0)

            if self.cache_dict is not None:
                self.cache_dict[index] = vector

            return np.array(vector)

        elif self.use_server == True: 
            print(f"... Fetching text {a} ...", end="\r")
            if a not in WORD_CACHE:
                req = requests.get(f"{VECTOR_SERVER}/transform/{a}")
                vec = np.array(json.loads(req.text))
                WORD_CACHE[a] = vec 
                return vec
            else:
                return WORD_CACHE[a]

        elif self.use_cache == True: 
            context, subcontext, name = self.get_dataset_info()
            column = self.column
            cached = VECTOR_CACHE[f"{context}|{subcontext}|{name}|{column}"]
            return cached[index]
          

    # ======================================================================== #

    def transform(self, X, y = None):
        vectors = []

        column = self.column

        if type(X) is pd.DataFrame:
            X = pd.Series(X[column])

        if self.use_cache:
            context, subcontext, name = self.get_dataset_info()
            if f"{context}|{subcontext}|{name}|{column}" not in VECTOR_CACHE:
                self.load_cache()
        
        if self.verbose: 
            print()

        i = 0 
        n = len(X)
        for index, item in X.items():
            a = item

            if self.verbose:
                print(f"... Preprocessing {i + 1} of {n} ...", end="\r")

            v = 0
            if type(a) is float or len(a) == 0:
                v = np.nan * np.ones(shape=(self.dims,))
            else:
                v = self.preprocess_text(a, index)
                if v.shape != (self.dims, ):
                    v = np.nan * np.ones(shape=(self.dims,))
                assert v.shape == (self.dims, ), f"v.shape: {v.shape}"
                
            vectors.append(v)

            i += 1

        vectors = np.array(vectors)

        print() 

        return vectors
