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

from loaders.datasets import *
# =========================================================================== #

models = pickle.load(open("model.dl.pickle", "rb"))
text = "i visited bagasbas beach, mount mayon and mount isarog in bicol" 

classes = []
probas = []

for model in models: 
    province = model.classes_[1]
    classes.append(province) 
    proba = model.predict_proba(pd.DataFrame({ "text" : [text] }))[0]
    print(province, proba[1])
    probas.append(proba[1])

plt.figure(figsize=(10, 50))
plt.barh(classes, stats.zscore(probas)) 
plt.savefig("proba.png")