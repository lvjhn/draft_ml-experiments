# =========================================================================== #
# :: COMMON HELPERS
#
# :: Description ::
#
# Define common helper functions that are used throughout the project.
#
# =========================================================================== #

# ==== COMMON IMPORTS ======================================================= #
from core.helpers.major_imports import *
from core.helpers.paths import *
# =========================================================================== #

# ==== NATIVE IMPORTS ======================================================= #
# =========================================================================== #

# ==== PROJECT IMPORTS ====================================================== #
from core.helpers.classifiers import *
from core.helpers.regressors import *
from core.helpers.common_imports import * 
# =========================================================================== #

class StatWrappers: 
    
    def mean(X): 
        return sum(X) / len(X)
    
    def mode(X): 
        return stats.mode(X) 

    def median(X): 
        return np.median(X) 

    def min(X): 
        return np.min(X) 

    def max(X):
        return np.max(X)
    
    def range(X):
        return np.max(X) - np.min(X)

    def std(X): 
        return np.std(X) 
    
    def var(X): 
        return np.var(X) 

    def cv(X):
        return stats.variation(X) 
    
    def kurtosis(X): 
        return stats.kurtosis(X) 

    def skew(X): 
        return stats.skew(X)

    def normality_sw(X): 
        if len(X) > 3 and not pd.Series(X).isnull().any().any():
            return float(stats.shapiro(X)[1])
        else: 
            return "N/A"

    def normality_ks(X): 
        if not pd.Series(X).isnull().any().any():
            return float(stats.kstest(X, "norm")[1])
        else: 
            return "N/A"

    def percentile(X, i): 
        return np.percentile(X, i)
