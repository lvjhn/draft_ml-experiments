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
from core.helpers.wrappers import *
# =========================================================================== #

# ==== NATIVE IMPORTS ======================================================= #
# =========================================================================== #

# ==== PROJECT IMPORTS ====================================================== #
from core.helpers.classifiers import *
from core.helpers.regressors import *
from core.helpers.common_imports import * 
# =========================================================================== #

class Common:
    def norm_path(path):
        """ 
            Replaces multiple slashes (/) in path.
            
            Parameters: 
                path - the path to handle
        """ 
        return re.sub(r"/+", "/", path)

    def detect_type(column, values): 
        """ 
            Detect types of a list/array of values. 

            Parameters: 
                values - the values to detect the type of 
            Returns: 
                type - the type of the values 
                        ("binary", "ordinal", "categorical",
                        "numeric_non_normal", "numeric_normal", ...)
        """ 
        import scipy 

        if type(values) is scipy.sparse.csr_matrix:
            return "csr_matrix", values
        
        # print(column, values)

        # detect if binary
        is_binary = Common.check_binary(values) 
        if is_binary: 
            if None in values:
                return "binary.with_null", values

            return "binary", values 

        # detect if numeric (attempts)
        is_numeric = Common.check_numeric(values)
        if is_numeric: 
            values = [Common.try_cast_to_float(x) for x in values]
            mean = sum(x for x in values if x != None) / len(values)
            values_ = []
            for i in range(len(values)):
                if values[i] == None:
                    values_.append(mean)
                else:
                    values_.append(values[i]) 
            values = values_

            if Common.is_normal(values):
                return "numeric.normal", values 
            else: 
                return "numeric.non_normal", values 

        # else, return categorical 
        if None in values:
            return "categorical.with_null", values

        return "categorical", values  


    def check_binary(values): 
        """ 
            Checks if a list/array of values is binary.

            Parameters: 
                values - the values to check if binary
            Returns: 
                is_numeric - whether the values are binary or not
        """ 
        set_values = set(values)
        
        if len(set_values) == 2: 
            return True 

        return False 

    def check_numeric(values): 
        """ 
            Checks if a list/array of values is numeric.
        `   
            Parameters: 
                values - the values to check if numeric
            Returns: 
                is_numeric - whether the values are numeric or not
        """ 

        
        # scoring
        valid_attempts = 0 
        total_attempts = len(values) 

        # find score
        for i in range(len(values)): 
            try:
                float(values[i])
                valid_attempts += 1
            except:
                pass 


        # check score 
        if valid_attempts / total_attempts >= 0.8: 
            return True 
        else: 
            return False 
    
    def try_cast_to_float(x): 
        """ 
            Attempts to cast a value to float, if it can't, returns none. 

            Parameters: 
                x - the value to attempt to cast to float
            Returns: 
                value - float version of x if possible else None
        """ 
        if x == "": 
            return None 
        try: 
            return float(x)
        except:
            return None 

    def normality_p_value(values): 
        """ 
            Computes the p-vaue conditionally. If the length of values 
            is < 5000, uses Shapiro-Wilk test. Else, uses Kolmogorov-Smirnov
            test. 

            Parameters: 
                values - the values to compute the p-value of 
            Returns: 
                p_value - the p-value of the values based on the normality test used
        """ 
        if len(values) < 5000: 
            return stats.shapiro(values[1])
        else: 
            return stats.kstest(values, "norm")[1]

    def is_normal(values): 
        """ 
            Checks if the given values are normal using three possible tests:
            Shapiro-Wilk, Anderson-Darling, and Kolmogorov-Smirnov. Failure of one
            means that the values are non-normal.

            Source:  
                https://machinelearningmastery.com
                    /a-gentle-introduction-to-normality-tests-in-python/
            
            Parameters:
                values - the values to check whether if normal or not
            Returns: 
                is_normal - whether the values are normal or not
        """ 
        if len(values) < 5000: 
            sw_normal      = Common.check_if_normal_sw(values)
            ks_normal      = Common.check_if_normal_ks(values)
            ad_normal      = Common.check_if_normal_ad(values) 
            return sw_normal and ks_normal and ad_normal
        else: 
            ks_normal      = Common.check_if_normal_ks(values)
            ad_normal      = Common.check_if_normal_ad(values) 
            return ks_normal and ad_normal

    def check_if_normal_sw(values):
        """ 
            Checks if the values provided are normal using Shapiro-Wilk
            test.

            Parameters: 
                values - the values to check the normality of
            Returns: 
                is_normal - whether the values are normal or not
        """ 
        if len(values) > 3:
            pvalue = stats.shapiro(values)[1] 
            return pvalue >= 0.05 
        else: 
            return False

    def check_if_normal_ks(values): 
        """ 
            Checks if the values provided are normal using Kolmogorov-Smirnov
            test.

            Parameters: 
                values - the values to check the normality of
            Returns: 
                is_normal - whether the values are normal or not
        """ 
        pvalue = stats.kstest(values, "norm")[1]
        return pvalue >= 0.05

    def check_if_normal_ad(values): 
        """ 
            Checks if the values provided are normal using Anderson-Darling
            test.

            Parameters: 
                values - the values to check the normality of
            Returns: 
                is_normal - whether the values are normal or not
        """
        result = stats.anderson(values)
        if result.statistic < result.critical_values[2]:
            return True 
        else: 
            return False

    def drop_columns(df, columns): 
        """ 
            Drop columns from dataframe individually.

            Parameters: 
                df - the dataframe to consider 
                columns - the columns to drop
        """ 
        for column in columns:
            df = df.drop(column, axis=1)
        return df


    def format_json(obj):
        '''
            Formats an object for JSON stringification
            
            Parameters:
                obj - object to stringify
            Returns:
                string - the stringified version of the object
        '''
        if type(obj) is np.ma.core.MaskedArray:
            return obj.tolist()
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        else:
            return str(obj)

    def get_value_counts(values): 
        """ 
            Gets the tally of percentage counts of values 
            as dictionary. 

            Parameters: 
                values - the values to tally
            Returns: 
                value_counts - the tally of counts
        """ 
        value_counts = Counter(values)
        return value_counts

    def get_value_percs(values):
        """ 
            Gets the tally of percentage counts of values 
            as dictionary. 

            Parameters: 
                values - the values to tally
            Returns: 
                value_counts - the tally of percentages
        """ 
        value_counts = Common.get_value_counts(values)
        total = sum(value_counts.values())
        for key in value_counts:
            value_counts[key] = value_counts[key] / total 
        return value_counts

    def get_sorted_value_counts(values):
        """ 
            Gets the tally of counts of values 
            as dictionary, sorted by value.

            Parameters: 
                values - the values to tally
            Returns: 
                value_counts - the tally of percentages
        """ 
        value_counts = Common.get_value_counts(values)
        value_counts = \
            dict(sorted(value_counts.items(), key=lambda item: -item[1]))
        return value_counts

    def get_sorted_value_percs(values):
        """ 
            Gets the tally of percentage counts of values 
            as dictionary, sorted by value.

            Parameters: 
                values - the values to tally
            Returns: 
                value_counts - the tally of percentages
        """ 
        value_counts = Common.get_value_percs(values)
        value_counts = \
            dict(sorted(value_counts.items(), key=lambda item: -item[1]))
        return value_counts

    def get_sorted_value_counts_by_key(values):
        """ 
            Gets the tally of counts of values 
            as dictionary, sorted by key.

            Parameters: 
                values - the values to tally
            Returns: 
                value_counts - the tally of percentages
        """ 
        value_counts = Common.get_value_counts(values)
        value_counts = \
            dict(sorted(value_counts.items(), key=lambda item: -item[0]))
        return value_counts

    def get_sorted_value_percs_by_key(values):
        """ 
            Gets the tally of percentage counts of values 
            as dictionary, sorted by value.

            Parameters: 
                values - the values to tally
            Returns: 
                value_counts - the tally of percentages
        """ 
        value_counts = Common.get_value_percs(values)
        value_counts = \
            dict(sorted(value_counts.items(), key=lambda item: -item[0]))
        return value_counts

    def get_unique_characters(texts):
        """ 
            Counts the unique no. of characters in the texts.

            Parameter: 
                texts - array of texts/strings
            Returns: 
                characters - set of unique characters found in texts
        """ 
        characters = set()
        for text in texts:
            for c in list(text):
                characters.add(c) 
        return characters

    def get_character_counts(texts):
        """ 
            Gets the character count of each character in texts.

            Parameter: 
                texts - array of texts/strings 
            Returns: 
                character_counts - 
                    the tally of occurency of each 
                    character in texts
        """ 
        character_counts = {}
        for text in texts:
            for c in list(text):
                if c not in character_counts: 
                    character_counts[c] = 0
                character_counts[c] += 1
        return character_counts 


    def get_sorted_character_counts(texts):
        """ 
            Gets the character count of each character in texts,
            sorted by value.

            Parameter: 
                texts - array of texts/strings 
            Returns: 
                character_counts - 
                    the tally of occurency of each 
                    character in texts (sorted by value)
        """ 
        character_counts = Common.get_character_counts(texts)
        character_counts = \
            dict(sorted(character_counts.items(), key=lambda item: -item[1]))
        return character_counts


    def get_summary_stats(values):
        """ 
            Gets summary statistics from a series of values.

            Parameters; 
                values - array/list of numeric values 
            Returns: 
                stats - summary statistics about `values`
        """  
        return {
            "count" : \
                len(values),
            "mean" : \
                float(StatWrappers.mean(values)), 
            "mean_count" : \
                float(list(values).count(StatWrappers.median(values))), 
            "mode" : \
                float(StatWrappers.mode(values)[0]),
            "mode_count" : \
                float(list(values).count(StatWrappers.mode(values)[1])),
            "median" : \
                float(StatWrappers.median(values)), 
            "median_count" : \
                float(list(values).count(StatWrappers.median(values))), 
            "min" : \
                float(StatWrappers.min(values)),
            "min_count" : \
                float(list(values).count(StatWrappers.min(values))),
            "max" : \
                float(StatWrappers.max(values)),
            "max_count" : \
                float(list(values).count(StatWrappers.max(values))),
            "range" : \
                float(StatWrappers.range(values)),
            "std" : \
                float(StatWrappers.std(values)),
            "var" : \
                float(StatWrappers.var(values)), 
            "cv" : \
                float(StatWrappers.cv(values)), 
            "kurtosis" : \
                float(StatWrappers.kurtosis(values)), 
            "skewness" : \
                float(StatWrappers.skew(values)), 
            "normality_sw" : \
                StatWrappers.normality_sw(values), 
            "normality_ks" : \
                StatWrappers.normality_ks(values), 
            "is_normal" : \
                True if Common.is_normal(values) else False,
            "skewness" : \
                float(StatWrappers.skew(values)), 
            "percentile_1" : \
                float(StatWrappers.percentile(values, 1)),
            "percentile_5" : \
                float(StatWrappers.percentile(values, 5)),
            "percentile_10" : \
                float(StatWrappers.percentile(values, 10)),
            "percentile_20" : \
                float(StatWrappers.percentile(values, 20)),
            "percentile_25" : \
                float(StatWrappers.percentile(values, 25)),
            "percentile_50" : \
                float(StatWrappers.percentile(values, 50)),
            "percentile_75" : \
                float(StatWrappers.percentile(values, 75)),
            "percentile_80" : \
                float(StatWrappers.percentile(values, 80)),
            "percentile_90" : \
                float(StatWrappers.percentile(values, 90)),
            "percentile_95" : \
                float(StatWrappers.percentile(values, 95)),
            "percentile_99" : \
                float(StatWrappers.percentile(values, 99))
        }

    def random_color():
        """ 
            Returns a random color.

            Returns:
                color - random color (HSV)
        """ 
        return np.random.rand(3,)

    def array_add(a, b):
        """ 
            Adds the elements of a 1D array in a pairwise manner.

            Parameters: 
                a - the first array
                b - the second array
            Returns: 
                c - array formed by adding a and b
        """  
        c = [a[i] + b[i] for i in range(len(a))]
        return c

    def get_rank_matrix(matrix):
        """ 
            Gets the row-wise rank of each item in a row of
            the parameter `matrix`.

            Parameters: 
                matrix - the matrix to compute the row-wise rank data of
            Returns: 
                ranked_matrix - matrix with row-wise ranking for each row
        """ 
        matrix = np.array(matrix)
        rank_matrix = []
        for i in range(matrix.shape[0]):
            rank_matrix.append(stats.rankdata(matrix[i])) 
        return np.array(rank_matrix)


    def heatmap(
        data = None, 
        columns = None, 
        rows = None,
        title = None,
        vmin = 0,
        vmax = 1,
        xlabel = "X",
        ylabel = "Y"
    ):
        """ 
            A shortcut to create a heatmap. 

            Parameter:  
                data - the data to tabulate 
                columns - column tickers
                rows - row tickers 
                title - title of the plot 
                vmin - min. value of the color scale
                vmax - max. value of the color scale 
                xlabel - label for x axis 
                ylabel - label for y axis 

        """ 
        plt.title("Correlation Matrix (Correlation Coefficient)")
        data = np.array(data)

        xaxis = np.arange(len(columns))
        yaxis = np.arange(len(rows))
        
        fig = plt.figure(
            figsize=(
                10 + len(columns) / 10, 
                10 + len(rows) / 10
            )
        )
        ax = fig.add_subplot(111)
        
        ax.set_title(title)

        cax = ax.matshow(data, interpolation='nearest', vmin=vmin, vmax=vmax)
        
        for (x, y), value in np.ndenumerate(data):
            plt.text(y, x, f"{value:.2f}", va="center", ha="center")
        
        fig.colorbar(cax)

        ax.set_xticks(xaxis, xaxis, rotation=90)
        ax.set_yticks(yaxis)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        ax.set_xticklabels(columns)
        ax.set_yticklabels(rows)

        return fig, ax 

    def to_table(X, Y, S = 10): 
        """ 
            Converts two columns X, Y to a discretized table
            format. 

            Parameters: 
                X - X "coordinates" in the grid 
                Y - Y "coordinates" in the grid 
                S - size of the grid
        """ 
        table = np.zeros((S + 1, S + 1))

        for i in range(len(X)):
            x_ = X[i]
            y_ = Y[i]

            if math.isnan(x_):
                x_ = 0
            if math.isnan(y_):
                y_ = 0

            if S == 10:   
                x_ = round(x_, 1)
                y_ = round(y_, 1)
                x_ *= 10
                y_ *= 10
                x_ = int(x_)
                y_ = int(y_)
            else: 
                x_ *= 100
                y_ *= 100
                x_ = int(x_)
                y_ = int(y_)
                x_ = x_ // (100 // S)
                y_ = y_ // (100 // S)
        
            table[y_][x_] += 1

        return table


    def table_max(table):
        """ 
            Gets the maximum value in a table. 

            Parameters:
                table - table to select the max. value from 
            Returns: 
                max_ - the max value in the table
        """ 
        max_ = float('-inf')
        for x in table:
            for y in x:
                if y > max_:
                    max_ = y
        return max_


    def table_min(table):
        """ 
            Gets the minimum value in a table. 

            Parameters:
                table - table to select the min. value from 
            Returns: 
                min_ - the min value in the table
        """ 
        min_ = float('inf')
        for x in table:
            for y in x:
                if y < min_:
                    min_ = y
        return min_


    def table_argmax(table):
        """ 
            Gets the maximum value in a table. 

            Parameters:
                table - table to select the max. value from 
            Returns: 
                max_ - the max value in the table
        """ 
        max_ = float('-inf')
        index = [None, None]
        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] > max_:
                    max_ = table[i][j]
                    index[0] = i
                    index[1] = j
        return tuple(index)


    def table_argmin(table):
        """ 
            Gets the minimum value in a table. 

            Parameters:
                table - table to select the min. value from 
            Returns: 
                min_ - the min value in the table
        """ 
        min_ = float('inf')
        index = [None, None]
        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] < min_:
                    min_ = table[i][j]
                    index[0] = i
                    index[1] = j
        return tuple(index)

    def combine_append(context, subcontext): 
        """ 
            Combines/appends subcontext into context.
        """ 
        for key in subcontext:
            if key not in context:
                context[key] = []
            context[key].append(subcontext[key])
        
    def get_hash(df): 
        """ 
            Gets hash of dataframe.
    """ 
        return joblib.hash(df)

    def print_dict(dict_, indent="d"):
        for key in dict_: 
            print(f"{indent}{key} = {dict_[key]}")

    def print_conf_matrix(labels, matrix): 
        table = []
        for i in range(matrix.shape[0]):
            row = matrix[i, :]
            row = [labels[i], *row]
            table.append(row)
        print(tabulate(table, headers=["", *labels]))

    def percentile_tuple(list_, percentile, index):
        N = len(list_)
        P = (N - 1) * (percentile / 100)
        if P % 1 == 0:
            return list_[int(P)][index]
        else:
            a = math.floor(P)
            b = math.ceil(P)
            return (list_[a][index] + list_[b][index]) / 2

    def patch_with_length(dataset): 
        df = dataset.state["df"]
        texts = dataset.state["grouped_types"]["text"]

        def compute_length(x):
            if type(x) is str: 
                return len(x)
            else: 
                return 0

        for text_column in texts: 
            df[f"{text_column} | Length"] = \
                df[text_column].apply(compute_length)

    def get_name_field(group): 
        if group == "island-groups": 
            return "Island Group"
        elif group == "regions": 
            return "Region"
        elif group == "provinces": 
            return "Province"
        elif group == "municities": 
            return "Municity" 
        else:
            raise Exception(f"Unknown group: {group}")