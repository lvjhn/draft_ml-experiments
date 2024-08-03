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
# =========================================================================== #

class Logger: 
    def __init__(
        self, 
        folder = "./logs/common/", 
        file = "main.log",
    ): 
        """ 
            Creates a logger object that writes to log files.

            Parameters: 
                folder 
                    - the folder where to write the log 
                
                file 
                    - name of the log file 
        """ 


        self.params = {
            "folder" : folder,
            "file"   : file
        }

        if not os.path.exists(self.path()):
            self.clear()

    ###########################################################################

    def path(self):
        return self.params["folder"] + "/" + self.params["file"]

    ###########################################################################

    def clear(self): 
        file_path = self.path()
        folder_path = "/".join(file_path.split("/")[:-1])
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path, exist_ok=True)
        return open(self.path(), "w").write("")

    ###########################################################################

    def file(self):
        return open(self.path() + ".txt", "a")

    ###########################################################################
                                                                       
    def divider(self): 
        return self.file().write("\n" + "=" * 80 + "\n") 

    ###########################################################################

    def new_run(self): 
        self.divider()
        current_datetime = \
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %z")    
        self.file().write("RUN-TIME : " + current_datetime + "\n")
        self.file().write("-" * 80 + "\n")

    ###########################################################################

    def remove(self): 
        return os.unlink(self.path())

    ###########################################################################
    
    def log_print(self, *args, print_=True, with_date=True):
        message = " ".join(args)
        if print_: 
            print(message)
        
        date = datetime.datetime.now()
        if with_date:
            self.file().write(f"[{date}] : {message} \n")
        else:
            self.file().write(message + "\n")

        return message