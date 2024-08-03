# =========================================================================== #
# :: KDAE
#
# :: Description ::
#   Keypoint Distance Embedding
#   - gets distances to keypoints and uses it to augment the feature space -
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


class KDAE(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        key_points = "label-centroids",
        agmv = None,
        distance = "cosine", 
        zscore = False,
        minmax = False,
        log = False, 
        with_main = True,
        with_dists = True,
        with_ranks = False,
        n_keypoints = 10,
        dims = 50,
        model = None,
        verbose = True
    ): 
        # define key points 
        self.key_points = key_points
        
        # define model 
        self.model = model

        # define model 
        self.agmv = agmv

        # define dims 
        self.dims = dims

        # define distance method 
        self.distance = distance

        # n keypoints 
        self.n_keypoints = n_keypoints

        # define other settings 
        self.log = log 
        self.zscore = zscore
        self.minmax = minmax

        self.with_dists = with_dists 
        self.with_ranks = with_ranks
        self.with_main = with_main

        # verbose 
        self.verbose = verbose

        # keypoints 
        self._key_points = key_points

    def fit_to_label_centroids(self, X, y = None):
        if self.model: 
            X_train  = self.model.X_train.iloc[:]
            y_train  = self.model.y_train.iloc[:]

            Xy       = pd.merge(
                X_train, y_train, 
                left_index=True, right_index=True
            )
            Xy_g     = Xy.groupby(by=Xy.columns[-1])

            centroids = {}

            for label, items in Xy_g:
                XV = self.agmv.transform(items[Xy.columns[0]])
                XV_mean = np.mean(XV, axis=0)
                centroids[label] = XV_mean 

            self._key_points = centroids
        
    def fit_to_random(self, X, y = None):
        key_points = {}

        for i in range(self.n_keypoints):
            key_points[i] = np.random.rand(25)

        self._key_points = key_points 

    def fit(self, X, y = None):      
        if self.key_points == "label-centroids":
            self.fit_to_label_centroids(X, y)

        elif self.key_points == "random":
            self.fit_to_random(X, y)

        return self
    
    def transform(self, X, y = None):
        distance_ = self.distance
        key_points = self._key_points.values()

        aug  = [] 
        for i in range(X.shape[0]): 
            V = X[i, :]
            aug_i = [] 

            for key_point in key_points:
                dist = None

                if distance_ == "cosine":
                    dist = distance.cosine(V, key_point)
                elif distance_ == "euclidean":
                    dist = distance.eucliean(V, key_point)
                elif distance_ == "manhattan":
                    dist = distance.cityblock(V, key_point)
                else:
                    raise Exception(f"Unknown distance method : {dist}")

                if self.log:
                    dist = math.log(dist) 
                
                aug_i.append(dist)

            if self.zscore:
                aug_i = stats.zscore(aug_i)
            if self.minmax:
                aug_i = MinMaxScaler().fit_transform([aug_i])[0]

            final = np.array([])

            if self.with_dists:
                aug_i = np.concatenate((final, aug_i))
            if self.with_ranks: 
                aug_i = np.concatenate((final, stats.rankdata(aug_i)))

            aug.append(aug_i)

        aug = np.array(aug)
        
        if self.with_main:
            X = np.concatenate((X, aug), axis=1)
        else: 
            X = aug

        return X
