import core.helpers as helpers

class Scraper:
    def __init__(
        client = None,
        download_outdir = None,
        output_filename = lambda p, t, i: f"{t}.html"
    ):
        # extract arguments 
        self.client = client
        self.download_outdir = download_outdir
        self.output_filename = output_filename

        # set download dir of client 
        self.client.download_outdir = self.download_outdir  

        # get list items to scrape
        self.items = [] 

    def add(self, tail): 
        self.items.append(tail)

    def add_multi(self, tails): 
        for tail in tails:
            self.items.append(tail)

    def scrape(self, verbose=False):
        client = self.client
        i = 0 
        n = len(self.items)

        for item in self.items: 
            verbose and print(f"@ Downloading [{item}] ({i + 1} of {n})")
            prefix = client.prefix 
            filename = self.output_filename(prefix, item, i)
            client.download(item, filename)
            i += 1