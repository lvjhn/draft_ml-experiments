import re 

from setup.settings import CONTEXT
import core.helpers as helpers

from core.wiki_article.extractor import Extractor

from bs4 import BeautifulSoup

class Article:
    def __init__(
        self, 
        article = None,
        folder = f"./data/locations/{CONTEXT}/wiki-articles/raw/main/"
    ):
        # dependency injection 
        self.Extractor = Extractor

        # extract arguments 
        self.article = article 
        self.folder = folder
        
        # load article content
        self.content = self.load_content()

        # load extractor 
        self.extractor = self.load_extractor()

    def load_content(self):
        article_path = self.folder + "/" + self.article + ".html"
        article_path = re.sub("/+", "/", article_path)
        content = open(article_path, "r").read() 
        return content

    def load_extractor(self): 
        return self.Extractor(self.content)

    def get_headers(self, level): 
        headers = self.extractor._.select(f"#content h{level}")
        headers = [
            header.get_text() 
            for header in headers
        ] 
        headers = [
            self.Extractor.normalize(h, remove_brackets=True) 
            for h in headers
        ]
        return headers

    def introduction(self): 
        return self.extractor._.select("#mw-content-text")[0]

    def top_level_sections(self): 
        current = self.extractor._.select("#content h2")[0]
        
        contents = {}
        current_section = ""

        while current != None:
            # set-up header
            if current.name == "h2": 
                normal_name = \
                    self.Extractor.normalize(
                        current.get_text(), 
                        remove_brackets=True
                    )
                contents[normal_name] = ""
                current_section = normal_name

            # append content 
            contents[normal_name] += str(current)
            
            # get next sibling
            current = current.findNextSibling()

            # combine contents 
            if current is None or current.name == "h2": 
                html = ( 
                    "<div class='section'>" +
                    str(contents[current_section]) +
                    "</div>"
                )
                contents[current_section] = \
                    BeautifulSoup(html, "html.parser")

        return contents

    def content_tree(self): 
        root = self.extractor._.select(".vector-toc-contents")[0]

        tree = {}

        def visit_list(base, ul): 
            items = ul.findChildren("li")

            for item in items: 
                # extract label
                label = item.select("a > div > span")
                
                if len(label) < 2: 
                    continue

                label = label[1].get_text() 
                base[label] = {}

                # visit subtree
                sublist = item.select("ul")
                if len(sublist) == 0:
                    continue
                visit_list(base[label], sublist[0])

        visit_list(tree, root)
        
        return tree
        
    def subsection(self, accessor): 
        start_level = 1
        n_levels = len(accessor)
        subcontext = None
        base_level = 0

        # find base element
        for i in range(n_levels): 
            level = start_level + i + 1
            access_item = accessor[i] 
            subcontext = \
                self.extractor\
                    .select_filtered(
                        f"h{level}", 
                        lambda x, h, t : access_item in t
                    )
            base_level = level
            
        # get content starting from base element
        current = subcontext.findNextSibling()
        html = "<div class='subcontent'>"
        while current is not None and current.name != f"h{base_level}":
            html += str(current)
            current = current.findNextSibling()
        html += "</div>"
        element = BeautifulSoup(html, "html.parser")
        return element

    def notes(self):
        references = self.extractor._.select(
            '.reflist.reflist-lower-alpha ' +
            '[id^="cite_note"]'
        ) 
        return [x for x in references]

    def references(self):
        references = self.extractor._.select(
            '.mw-references-wrap.mw-references-columns ' +
            '[id^="cite_note"]'
        ) 
        return [x for x in references]