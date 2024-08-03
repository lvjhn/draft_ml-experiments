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
from sklearn.base import BaseEstimator, TransformerMixin
# =========================================================================== #

# ==== PROJECT IMPORTS ====================================================== #
from core.helpers.classifiers import *
from core.helpers.regressors import *
from core.helpers.common_imports import * 
from core.helpers.project_imports import * 

from .extra_transforms import *
# =========================================================================== #

class Patcher:
    def __init__(self, name, BasePipeline, patches = None, *args, **kwargs):
        self.pipeline        = None 
        self.base_pipeline   = None
        self.name            = name
        self.BasePipeline    = BasePipeline
        self.patches         = patches 
        self.apply_patches()

    def apply_patches(self): 
        PatchedClass = self.BasePipeline
        patches = self.patches 
        for method in patches:    
            setattr(PatchedClass, method, patches[method]) 
        self.base_pipeline = PatchedClass()
      
    def create_pipeline(self, *args, **kwargs): 
        self.base_pipeline.create_pipeline(*args, **kwargs)
        self.pipeline = self.base_pipeline.pipeline
        self.input_size = self.base_pipeline.input_size
         