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

class Lemmatizer(BaseEstimator, TransformerMixin):
    def __init__(
        self
    ): 
       pass

    # ======================================================================== #

    def fit(self, X, y = None): 
        return self

    # ======================================================================== #

    def transform(self, X, y = None):
        from gensim.utils import simple_preprocess 
        from nltk.stem import PorterStemmer
        
        ps = PorterStemmer()

        if type(X) is pd.DataFrame:
            X = pd.Series(X[X.columns[0]])

        for index, item in X.items():
            words = simple_preprocess(item)
            X.at[index] = " ".join([ps.stem(word) for word in words])

        return X
