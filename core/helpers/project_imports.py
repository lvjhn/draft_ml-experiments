#
# PROJECT-LEVEL IMPORTS
#
from core.datasets.dataset import Dataset
from core.datasets.labeled_dataset import LabeledDataset
from core.reporting.logger import Logger
from setup.transforms.agmv import AGMV

#
# CLASSIFIERS
#
from setup.pipelines.base.classifiers.BL_C_DC import BL_C_DC

from setup.pipelines.base.classifiers.BL_C_KNN  import BL_C_KNN

from setup.pipelines.base.classifiers.BL_C_GNB  import BL_C_GNB
from setup.pipelines.base.classifiers.BL_C_MNB  import BL_C_MNB
from setup.pipelines.base.classifiers.BL_C_BNB  import BL_C_BNB
from setup.pipelines.base.classifiers.BL_C_CNB  import BL_C_CNB

from setup.pipelines.base.classifiers.BL_C_DTC  import BL_C_DTC
from setup.pipelines.base.classifiers.BL_C_RFC  import BL_C_RFC


from setup.pipelines.base.classifiers.BL_C_RC   import BL_C_RC
from setup.pipelines.base.classifiers.BL_C_LR   import BL_C_LR
from setup.pipelines.base.classifiers.BL_C_SGDC import BL_C_SGDC

from setup.pipelines.base.classifiers.BL_C_LSVC import BL_C_LSVC
from setup.pipelines.base.classifiers.BL_C_SVC  import BL_C_SVC

from setup.pipelines.base.classifiers.BL_C_LDA  import BL_C_LDA
from setup.pipelines.base.classifiers.BL_C_QDA  import BL_C_QDA

from setup.pipelines.base.classifiers.BL_C_MLPC import BL_C_MLPC

from setup.pipelines.custom.mlpc_sl.BL_C_MLPC_2 import BL_C_MLPC_2
from setup.pipelines.custom.mlpc_sl.BL_C_MLPC_3 import BL_C_MLPC_3
from setup.pipelines.custom.mlpc_sl.BL_C_MLPC_4 import BL_C_MLPC_4
from setup.pipelines.custom.mlpc_sl.BL_C_MLPC_5 import BL_C_MLPC_5
from setup.pipelines.custom.mlpc_sl.BL_C_MLPC_6 import BL_C_MLPC_6
from setup.pipelines.custom.mlpc_sl.BL_C_MLPC_7 import BL_C_MLPC_7
from setup.pipelines.custom.mlpc_sl.BL_C_MLPC_8 import BL_C_MLPC_8
from setup.pipelines.custom.mlpc_sl.BL_C_MLPC_9 import BL_C_MLPC_9
from setup.pipelines.custom.mlpc_sl.BL_C_MLPC_10 import BL_C_MLPC_10
from setup.pipelines.custom.mlpc_sl.BL_C_MLPC_20 import BL_C_MLPC_20
from setup.pipelines.custom.mlpc_sl.BL_C_MLPC_50 import BL_C_MLPC_50
from setup.pipelines.custom.mlpc_sl.BL_C_MLPC_80 import BL_C_MLPC_80

#
# REGRESSORS
#
from setup.pipelines.base.regressors.BL_R_DR import BL_R_DR

from setup.pipelines.base.regressors.BL_R_LR  import BL_R_LR
from setup.pipelines.base.regressors.BL_R_R  import BL_R_R
from setup.pipelines.base.regressors.BL_R_HR import BL_R_HR
from setup.pipelines.base.regressors.BL_R_L  import BL_R_L
from setup.pipelines.base.regressors.BL_R_EN  import BL_R_EN
from setup.pipelines.base.regressors.BL_R_Lars  import BL_R_Lars
from setup.pipelines.base.regressors.BL_R_LL import BL_R_LL
from setup.pipelines.base.regressors.BL_R_BR import BL_R_BR
from setup.pipelines.base.regressors.BL_R_SGDR import BL_R_SGDR

from setup.pipelines.base.regressors.BL_R_LSVR import BL_R_LSVR
from setup.pipelines.base.regressors.BL_R_SVR  import BL_R_SVR

from setup.pipelines.base.regressors.BL_R_DTR import BL_R_DTR
from setup.pipelines.base.regressors.BL_R_RFR import BL_R_RFR

from setup.pipelines.base.regressors.BL_R_KNR import BL_R_KNR

from setup.pipelines.base.regressors.BL_R_MLPR import BL_R_MLPR
