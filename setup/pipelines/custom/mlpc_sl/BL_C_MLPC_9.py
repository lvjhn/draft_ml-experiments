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
from setup.pipelines.base_pipeline import BasePipeline
# =========================================================================== #

# ==== PROJECT IMPORTS ====================================================== #
from core.helpers.classifiers import *
from core.helpers.regressors import *
from core.helpers.common_imports import * 
from core.helpers.project_imports import * 
# =========================================================================== #

class BL_C_MLPC_9(BasePipeline):
    def define_task_estimator(self): 
        return MLPClassifier(hidden_layer_sizes=(9,), random_state=RANDOM_STATE) 

    def define_normal_features(self): 
        return [StandardScaler()]

    def define_numeric_non_normal_features(self): 
        return [StandardScaler()]

    