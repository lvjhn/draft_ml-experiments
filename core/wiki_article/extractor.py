import core.helpers as helpers
from bs4 import BeautifulSoup
import re 

class Extractor:
    SkipItem    = dict()

    DIGITS      = set("0123456789")
    LETTERS     = set("abcdefghijklmnopqrstuvwxyz")
    SYMBOLS     = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    HYPHEN      = set("-")
    DASH        = set("-")
    APOSTROPHE  = set("'")
    PERIOD      = set(".")
    MONTHS      = {
        "Jan" : 1, 
        "Feb" : 2, 
        "Mar" : 3, 
        "Apr" : 4, 
        "May" : 5, 
        "Jul" : 6, 
        "Jun" : 7, 
        "Aug" : 8, 
        "Sep" : 9, 
        "Oct" : 10, 
        "Nov" : 11, 
        "Dec" : 12
    }


    FILTERS = {
        "digits" : DIGITS,
        "letters" : LETTERS, 
        "symbols" : SYMBOLS, 
        "hyphens" : HYPHEN,
        "dashes" : DASH, 
        "apostrophe" : APOSTROPHE, 
        "period" : PERIOD
    }

    def __init__(self, content, *args, **kwargs):
        # extract arguments 
        self.content = content
        self._       = BeautifulSoup(self.content, "html.parser")

    #
    # NORMALIZATION
    # 


    def normalize(text, **kwargs): 
        if kwargs.get("remove_brackets", True):
            text = re.sub(r"\[.*\]", "", text)
        
        if kwargs.get("remove_parentheses", False):
            text = re.sub(r"\(.*\)", "", text)
        
        if kwargs.get("remove_braces", False):
            text = re.sub(r"\{.*\}", "", text)
        
        if kwargs.get("remove_digits", False):
            text = re.sub(r"[0-9]", "", text)
        
        if kwargs.get("remove_symbols", False):
            for c in list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"):
                text = text.replace(c, "")

        # TO-TEST
        if kwargs.get("remove_hyphens", False):
           text = text.replace("-", " ")
        
        if kwargs.get("title_case", False):
            text = " ".join([
                x[0].upper() + x[1:] 
                for x in text.split(" ") if len(x) > 0
            ])
        
        if kwargs.get("kebab_case", False):
            text = "-".join([x for x in text.split(" ")])
        
        if kwargs.get("pascal_case", False):
            text = "".join([x[0].upper() + x[1:] for x in text.split(" ")])

        if kwargs.get("camel_case", False):
            text = "".join([x[0].upper() + x[1:] for x in text.split(" ")])
            text = text[0].lower() + text[1:]

        if kwargs.get("snake_case", False):
            text = "_".join([x for x in text.split(" ")])

        if kwargs.get("lower_case", False):
            text = text.lower()

        if kwargs.get("upper_case", False):
            text = text.upper()

        if kwargs.get("trim", False): 
            text = text.strip()


        return text

    def filter(text, filter_="", **kwargs): 
        if text is None:
            return None
        text = [x for x in list(text) if x in filter_]    
        text = "".join(text)
        return text

    def to_float(text, filter_="", **kwargs): 
        if text is None or text == "—" or text == "":
            return None
        text = text.replace("−", "-")
        return float(Extractor.filter(text, filter_="-1234567890."))

    def to_int(text, filter_="", **kwargs): 
        if text is None or text == "—":
            return None
        text = text.replace("−", "-")
        return int(Extractor.filter(text, filter_="-1234567890"))

    
    def deperc(text, filter_="", **kwargs): 
        if text is None:
            return None
        text = text.replace("−", "-")
        return float(Extractor.filter(text, filter_="-1234567890."))

    def first_or_null(pattern, text):
        res = re.findall(pattern, text)
        if len(res) == 0: 
            return None 
        return res[0]

    def all_or_null(pattern, text):
        res = re.findall(pattern, text) 
        if len(res) == 0: 
            return None 
        return res

    def remove_dot(x):
        x = re.sub("\xa0•", "", x)
        x = re.sub("\xa0", "", x)
        return x

    def area_split(df, field):
        df[f"{field}_km2"] = \
            df[field].apply(
                lambda x: 
                    Extractor.to_float(x.split("km2")[0].strip())
            )
        
        df[f"{field}_mi2"] = \
            df[field].apply(
                lambda x: 
                    Extractor.to_float(
                        Extractor.first_or_null(r"\((.*)\ssq\smi\)", x)
                    )
            )

        df = df.drop(field, axis=1)

        return df

    def density_split(df, field):
        df[f"{field}_km2"] = \
            df[field].apply(
                lambda x: 
                    Extractor.to_float(x.split("/km2")[0].strip())
            )

        df[f"{field}_mi2"] = \
            df[field].apply(
                lambda x: 
                    Extractor.to_float(
                        Extractor.first_or_null(r"\((.*)\/?\s?sq\smi\)", x)
                    )
            )

        df = df.drop(field, axis=1)

        return df 

    def date_split_item(item, version="v1"): 
        tokens = item.replace(",", "")
        tokens = tokens.split(" ") 
        if len(tokens) == 1: 
            return (None, None, item.strip())
        else: 
            return tuple([x.strip() for x in tokens])

    def date_split(df, field):
        df[f"{field}_date"] = \
            df[field].apply(
                lambda x: 
                    Extractor.date_split_item(x)
            )
            
        df[f"{field}_a"] = \
            df[f"{field}_date"].apply(lambda x: x[0])

        df[f"{field}_b"] = \
            df[f"{field}_date"].apply(lambda x: x[1])

        df[f"{field}_c"] = \
            df[f"{field}_date"].apply(lambda x: x[2])

        df = df.drop(field, axis=1)

        return df 

    def extract_table_links(self, filters, row_index):
        cols = self.extract_column(
            "table", 
            row_index,
            filter_=filters,
            each=lambda x, i: x
        )
        links = [
            col.select("a")[0]["href"] for col in cols 
            if len(col.select("a")) > 0
        ]
        return links

    def scale_words(y):
        y = y.replace("trillion", "T")
        y = y.replace("billion", "B")
        y = y.replace("million", "M")
        y = y.replace("million", "K")
        y = y.replace("hundred", "H")
        return y

    def scale_item(y):
        scale = 0
        
        if "trillion" in y: 
            scale = 1e12
        elif "billion" in y:
            scale = 1e9 
        elif "million" in y: 
            scale = 1e6 
        elif "thousand" in y: 
            scale = 1e3 
        
        y = re.sub(" (trillion|billion|million|thousand)", "", y)
        y = Extractor.to_float(y)
        y = y * scale

        return y

    def select_filtered(self, sel, filter_ = None, base = None):
        root = self._

        if type(sel) is not str: 
            return sel

        if base == None:
            base = self._ 
    
        if filter_ == None:
            el = base.select(sel)[0]
            return el
        else: 
            el = base.select(sel)
            for x in el:
                is_ok = filter_(x, str(x), x.get_text())
                if is_ok:
                    return x
            return None
        

    def from_headers(self, headers, dis="~!@#$%^&*()", header_index=0): 
        def inner(x, h, t):
            dis_    = dis not in t 

            headers_ = \
                self.extract_table_headers(
                    x, 
                    header_index=header_index
                )

            if headers_ is None: 
                return False

            results = [] 
            for header_a in headers: 
                result = False
                header_a = header_a.replace("\n", " ")
                for header_b in headers_:
                    header_b = header_b.replace("\n", " ")
                    if header_a in header_b: 
                        result = True 
                        break
                results.append(result)


            if dis_ and len(set(results)) == 1 and results[0] == True:
                return True
            else: 
                return False
            

        return inner


    #
    # TABLE
    # 

    def extract_row(
        self, 
        sel, 
        row_index, 
        filter_=None, 
        each=lambda x, i: x.get_text().strip(),
        base=None
    ): 
        table  = self.select_filtered(sel, filter_, base) 
        rows   = table.select("tbody > tr")
        row    = rows[row_index] 
        fields = row.select("td")
        res    = [each(fields[i], i) for i in range(len(fields))] 
        return res

    def extract_column(
        self, 
        sel, 
        col_index, 
        filter_=None, 
        each=lambda x, i: x.get_text().strip(),
        base=None
    ): 
        rows   = self.extract_table_body(
            sel, 
            filter_=filter_, 
            each=lambda x, i, j: x
        ) 
        fields = []
        for row in rows: 
            if len(row) > col_index:
                fields.append(row[col_index])
        res    = [each(fields[i], i) for i in range(len(fields))] 
        return res

    def extract_table_headers(
        self, 
        sel, 
        filter_=None, 
        each=lambda x, i: x.get_text().strip(),
        normalize=False,
        base=None,
        prenorm=False,
        header_index=0
    ): 
        table  = self.select_filtered(sel, filter_, base) 

        table_header = table.select("thead > tr")
        if len(table_header) == 0: 
            table_header = table.select("tbody > tr")
           
            if len(table_header) == 0:
                return None

        hrow   = table_header[header_index]
        fields = hrow.select("th")

        def prenorm(x):
            return (
                BeautifulSoup(str(x).replace("<br/>", " "), "html.parser")
            )

        res    = [each(prenorm(fields[i]), i) for i in range(len(fields))] 

        if normalize:
            res    = [
                self.normalize(x, lower_case=True, snake_case=True) 
                for x in res
            ]

        return res

    def extract_table_body(
        self, 
        sel, 
        filter_=None, 
        each=lambda x, i, j: x.get_text().strip(),
        base=None
    ): 
        table  = self.select_filtered(sel, filter_, base) 
        rows   = table.select("tbody > tr")
        res    = [] 

        def prenorm(x):
            norm = (
                BeautifulSoup(
                    str(x)
                        .replace("<br/>", " ")
                        .replace("</li>", "|</li>"), 
                    "html.parser"
                )
            )
            return norm
                 

        for i in range(len(rows)): 
            row = rows[i] 
            row_new = [] 
            fields = row.select("th, td", recursive=False) 
            for j in range(len(fields)):
                field = fields[j]
                item = each(prenorm(field), i, j) 
                row_new.append(item)
            
            res.append(row_new)

        return res

    def extract_pair(
        self, 
        field, 
        select=lambda x: Extractor.normalize(
            x.get_text(), 
            remove_brackets=True
        ),
        base=None
    ): 
        field  = self.select_filtered(
            "th", 
            lambda x, h, t: field in t,
            base
        )
        parent = field.parent
        value  = parent.select("td")[0]
        value  = select(value)  
        return value
        

    def extract_pairs_from_partition(
        self, 
        start_field, 
        select=lambda x, y, i: (
            x.get_text().strip().replace("•\xa0", ""),
            y.get_text().strip()
        ),
        base=None
    ): 
        start_field_  = \
            self.select_filtered(
                "th", 
                lambda x, h, t: start_field in t,
                base
            )

        current = start_field_.parent.findNextSibling() 
        
        pairs = []

        i = 0
        while True:
            x = current.select("th")
            y = current.select("td")

            if len(x) == 0 or len(y) == 0:
                current = current.findNextSibling()
                continue 

            x = x[0]
            y = y[0]

            item = select(x, y, i)
            
            pairs.append(item)
            
            if current.has_attr("class") and \
               "mergedbottomrow" in current["class"]:
                break

            current = current.findNextSibling()

            if current is None: 
                break 

            if len(current.select("td")) == 0 or \
                len(current.select("th")) == 0 or \
                (current.has_attr("class") and \
                "mergedtoprow" in current["class"]):
                break

            i += 1

        return pairs

    #
    # LIST
    # 
    def extract_list(
        self, 
        sel, 
        filter_=None,
        each=lambda x, i: x.get_text().strip(),
        base=None
    ): 
        if base == None:
            base = self._
        list_ = self.select_filtered(sel, filter_, base)
        items = list_.select("li", recursive=False) 
        res = []
        for i in range(len(items)): 
            item = each(items[i], i)
            res.append(item) 
        return res