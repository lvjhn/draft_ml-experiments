from sklearn.base import BaseEstimator, TransformerMixin
from core.helpers.common import Common
import pandas as pd
import scipy

class DropNonNumericTransformer:
    def __init__(self, *args, **kwargs):
        self.to_exclude = set()

    def fit(self, X, y = None): 
        for i in range(X.shape[1]): 

            if type(X[:, i]) is scipy.sparse.csr_matrix: 
                continue

            values = X[:, i]

            for value in values: 
                if type(value) is str:
                    self.to_exclude.add(i)
                    break
                try:
                    float(value)
                except:
                    self.to_exclude.add(i)
                    break
          
        return self

    def transform(self, X, y = None):
        indices = list(range(X.shape[1]))
        to_include = [] 
        for i in indices:
            if i not in self.to_exclude: 
                to_include.append(i) 

        return X[:, to_include]

    def fit_transform(self, X, y = None): 
        self.fit(X)
        return self.transform(X)

    def get_params(self, *args, **kwargs):
        return {}

class ReImputer(BaseEstimator, TransformerMixin): 
    def __init__(
        self,
        context = None,
        categorical = None, 
        numeric = None, 
        *args, 
        **kwargs
    ):
        self.context = context

        from sklearn.impute import SimpleImputer

        self.categorical = SimpleImputer(strategy="most_frequent")
        self.numeric = SimpleImputer(strategy="mean")

        self.categorical_indices = []
        self.numeric_indices = []

    def fit(self, X, y = None): 
        for i in range(X.shape[1]): 
            values = X[:, i]
            type_, values = Common.detect_type(i, values)
            if type_ == "binary" or type_ == "categorical": 
                self.categorical_indices.append(i) 
            else: 
                self.numeric_indices.append(i)
        return self

    def transform(self, X, y = None):
        from sklearn.compose import ColumnTransformer 

        transformer = ColumnTransformer(
            transformers = [
                (
                    "categorical", 
                    self.categorical, 
                    self.categorical_indices
                ),
                (
                    "numeric", 
                    self.numeric, 
                    self.numeric_indices
                )
            ]
        )

        transformer.fit(X, y)
        X = transformer.transform(X)

        return X

    def fit_transform(self, X, y = None): 
        self.fit(X)
        return self.transform(X)

    def get_params(self, *args, **kwargs):
        return {}

class ApplyVectorizer(BaseEstimator, TransformerMixin): 
    def __init__(
        self,
        vectorizer = None, 
        field = "Headline"
    ):
        self.vectorizer = vectorizer
        self.field = field

    def fit(self, X, y = None): 
        X = pd.Series(X[self.field])
        return self.vectorizer.fit(X)

    def transform(self, X, y = None):
        X = self.vectorizer.transform(pd.Series(X[self.field]))
        return X
