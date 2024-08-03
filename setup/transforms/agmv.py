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


class AGMV(BaseEstimator, TransformerMixin):
    CACHE = {}

    def __init__(
        self,
        model_name = "glove-twitter-25",
        model_obj = None,
        dims = 25,
        averaging = "word",
        verbose = True
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

        # load model if not yet in cache
        if model_name not in self.CACHE: 
            if self.verbose:
                print(f"===> AGMV : Loading model `{self.model_name}` <===")
            if model_obj is not None: 
                AGMV.CACHE[model_name] = model_obj 
            else:
                AGMV.CACHE[model_name] = api.load(model_name) 
        
        # make model object wide
        self.model = AGMV.CACHE[model_name]

    def vectorize_word(self, w):
        if w not in self.model:
            return self.model["unk"]
        return self.model[w]

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

    def fit(self, X, y = None): 
        return self

    def preprocess_text(self, a): 
        from nltk import tokenize

        vector           = None

        if self.averaging == "word":
            words       = tokenize.word_tokenize(a) 
            vector     = [self.vectorize_word(w) for w in words]

        elif self.averaging == "sentence":
            words      = tokenize.word_tokenize(a) 
            sentences  = tokenize.sent_tokenize(a) 
            sentences = [tokenize.word_tokenize(s) for s in sentences]
            vector    = [self.vectorize_sentence(s) for s in sentences]

        vector = np.mean(vector, axis=0)

        return np.array(vector)

    def transform(self, X, y = None):
        vectors = []

        if type(X) is pd.DataFrame:
            X = pd.Series(X[X.columns[0]])

        i = 0 
        n = len(X)
        for index, item in X.items():
            a = item

            if self.verbose:
                print(f"... Preprocessing {i + 1} of {n} ...", end="\r")

            v = 0
            if len(a) == 0:
                v = np.nan * np.ones(shape=(self.dims,))
            else:
                v = self.preprocess_text(a)
                if v.shape != (self.dims, ):
                    v = np.nan * np.ones(shape=(self.dims,))
                assert v.shape == (self.dims, ), f"v.shape: {v.shape}"
            vectors.append(v)

            i += 1

        vectors = np.array(vectors)

        print() 

        return vectors
