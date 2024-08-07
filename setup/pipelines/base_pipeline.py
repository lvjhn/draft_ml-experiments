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


from .extra_transforms import *
# =========================================================================== #

def inspect(X):
    print(X)
    quit()
    return X

class BasePipeline:
    def __init__(self): 
        self.pipeline = None 
        self.input_size = 0
        self.task_type = None
   
    # ----- DEFINE PIPELINE ----- #
    def create_pipeline(self, dataset, task_type):
        pipeline = None 
        
        self.task_type = task_type 
        self.dataset = dataset 

        # get grouped types 
        base_dataset = dataset.state["dataset"]
        grouped_types = base_dataset.state["grouped_types"]
        features = set(dataset.state["feature_columns"])
        
        cbos    = set() 
        cbos    = cbos.union(grouped_types["categorical"])
        cbos    = cbos.union(grouped_types["binary"])
        cbos    = cbos.union(grouped_types["ordinal"])

        texts   = set() 
        texts   = texts.union(grouped_types["text"])

        numerics = set() 
        numerics = texts.union(grouped_types["numeric.normal"])
        numerics = texts.union(grouped_types["numeric.non_normal"])

        non_others = numerics.union(texts).union(cbos)
        others     = features.difference(non_others)

        gtypes = grouped_types

        # create base imputers 
        base_imputers   = self.add_base_imputers(
            dataset, features, gtypes, cbos, numerics
        )
      
        # create column transformers
        numeric_trans   = \
            self.add_numeric_steps(dataset, numerics, features, gtypes)
        cbos_trans      = \
            self.add_cbos_steps(dataset, cbos, features, gtypes)
        texts_trans     = \
            self.add_texts_steps(dataset, texts, features, gtypes)
        others_trans    = \
            self.add_others_steps(dataset, others, features, gtypes)
        extra_cols_trans = \
            self.define_extra_column_transforms()

        transformers = numeric_trans + cbos_trans + texts_trans + others_trans 

        if extra_cols_trans:
            transformers += extra_cols_trans

        # create column transformers
        col_trans = ColumnTransformer(
            transformers = transformers,
            remainder = "drop"
        )

        # create reimputer
        reimputer = ReImputer()

        # create resampler
        resampler = self.define_resampling()

        # create column dropper 
        drop_cols = DropNonNumericTransformer()
        
       

        # create unsparser
        def unsparse(X):
            if task_estimator is GaussianNB: 
                return X.toarray()
            else:
                return X
        
        # # create input no. detector:
        # def input_size_detector(X):
        #     self.input_size = X.shape[1]
        #     return X


        # create column steps 
        steps = [
            ("col_trans", col_trans),
        ]

        # create overall steps
        steps += [
            ("reimputer", reimputer), 
        ]
        
        if task_type == "classification": 
            steps += [
                ("resampler", resampler)
            ]

        steps += [
            ("drop_cols", drop_cols),

        ]

        # add extra transformations
        extra_transforms = self.define_extra_transforms()
        if extra_transforms is not None: 
            steps = steps + extra_transforms

        # detect shape of inputs 
        # steps += [
        #     (
        #         "input_size_detector", 
        #         FunctionTransformer(input_size_detector)
        #     )
        # ]

        # add task estimator
        task_estimator = self.define_task_estimator()
        steps.append(("task_estimator", task_estimator))

        # create pipeline 
        pipeline = Pipeline(steps=steps)
        self.pipeline = pipeline 

        # transform targets 
        if task_type == "regression":
            pipeline = TransformedTargetRegressor(
                regressor=pipeline, 
                transformer=MinMaxScaler()
            )

        return pipeline 

    # ----- FILTER FEATURES ----- #
    def filter_features(self, columns, features): 
        return list(set(columns) & set(features))

    # ----- DEFINE IMPUTERS ----- #
    def add_base_imputers(self, dataset, features, gtypes, cbos, numerics): 
        return [
            (
                "impute:categorical", 
                self.define_categorical_imputations(), 
                self.filter_features(
                    gtypes["cbos"], features
                )
            ),
            (
                "impute:numeric.normal", 
                self.define_numeric_normal_imputations(), 
                self.filter_features(
                    gtypes["numeric.normal"], features
                )
            ),
            (
                "impute:numeric.non_normal", 
                self.define_numeric_non_normal_imputations(), 
                self.filter_features(
                    gtypes["numeric.non_normal"], features
                )
            )
        ]
        
    # ----- DEFINE NUMERIC FEATURES ----- #
    def add_numeric_steps(self, dataset, numerics, features, gtypes): 
        return [
            (
                "transform:numeric.normal", 
                make_pipeline(*self.define_numeric_normal_features()), 
                self.filter_features(
                    gtypes["numeric.normal"], features
                )
            ),
            (
                "transform:numeric.non_normal", 
                make_pipeline(*self.define_numeric_non_normal_features()), 
                self.filter_features(
                    gtypes["numeric.non_normal"], features
                )
            )
        ]
  

    # ----- DEFINE CBOS FEATURES ----- #
    def add_cbos_steps(self, dataset, cbos, features, gtypes): 
        return [
            (
                "transform:binary", 
                make_pipeline(*self.define_binary_features()), 
                self.filter_features(
                    gtypes["binary"], features
                )
            ),
            (
                "transform:categorical", 
                make_pipeline(*self.define_categorical_features()), 
                self.filter_features(
                    gtypes["categorical"], features
                )
            ),
            (
                "transform:ordinal", 
                make_pipeline(*self.define_ordinal_features()), 
                self.filter_features(
                    gtypes["ordinal"], features
                )
            )
        ]

    # ----- DEFINE TEXTS FEATURES ----- #
    def add_texts_steps(self, dataset, texts, features, gtypes): 
        steps = []

        for text_column in texts: 
            if text_column not in features: 
                continue
            else:
                transformers = self.define_text_features(text_column)

                steps.append(
                    (   
                        f"transform:text.{text_column}",
                        ApplyVectorizer(
                            vectorizer = make_pipeline(*transformers),
                            field = text_column
                        ),
                        [text_column]
                    )
                )

        
        return steps

    # ----- DEFINE OTHERS FEATURES ----- #
    def add_others_steps(self, dataset, others, features, gtypes): 
        steps = []

        for column in others: 
            if column not in features: 
                continue
            else:
                type_ = \
                    dataset.state["dataset"].state["types"][column]

                if hasattr(self, f"define_{type_}_features"):
                    steps.append(
                        (   
                            f"transform:{type_}.{column}",
                            make_pipeline(
                                *getattr(self, f"define_{type_}_features")()
                            ),
                            column
                        )
                    )

        return steps

    # ----- DEFINE TASK ESTIMATOR ----- #
    def define_task_estimator(self): 
        return DummyClassifier(strategy="uniform")

    # ----- DEFINE EXTRA COLUMN TRANSFORMS ----- #
    def define_extra_column_transforms(self): 
        return None

    # ----- DEFINE EXTRA TRANSFORMS ----- #
    def define_extra_transforms(self): 
        # return [("pca", PCA(n_components=6)), ("ss", StandardScaler())]
        return None 

    # ----- DEFINE IMPUTATIONS ----- # 
    def define_categorical_imputations(self): 
        return SimpleImputer(strategy="most_frequent") 
    
    def define_numeric_normal_imputations(self): 
        return SimpleImputer(strategy="mean") 
    
    def define_numeric_non_normal_imputations(self): 
        return SimpleImputer(strategy="median")    

    # ----- DEFINE RESAMPLING ----- # 
    def define_resampling(self): 
        return RandomOverSampler()

    # ----- DEFINE TYPE TRANSFORMATION ----- #

    def define_binary_features(self): 
        return [OneHotEncoder(handle_unknown="ignore")] 

    def define_categorical_features(self): 
        return [OneHotEncoder(handle_unknown="ignore")] 
    
    def define_ordinal_features(self): 
        return [
            OrdinalEncoder(
                handle_unknown='ignore', 
                unknown_value = np.nan
            )
        ] 

    def define_numeric_non_normal_features(self): 
        return [StandardScaler()]

    def define_numeric_normal_features(self): 
        return [StandardScaler()] 
    
    def define_numeric_with_null_features(self): 
        return [StandardScaler()] 

    def define_text_features(self, column): 
        
        # # --- define AGMV
        # agmv = AGMV(
        #     model_name = "glove-wiki-gigaword-50",
        #     dims = 50, 
        #     verbose = True,
        #     use_cache = False,
        #     pipeline = self,
        #     column = column,
        #     cache_dict = self.cache_dict,
        #     averaging = "word"
        # )

        # return [ agmv ]

        return [ 
            Lemmatizer(), 
            ApplyVectorizer(
                vectorizer=TfidfVectorizer(stop_words="english"), 
                field="text"
            ) 
        ]