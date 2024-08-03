cpulimit -l 40 -- python3 -m utils.precomp_vecs "kaggle|amananandrai|AG News Classification Dataset|ag-news.csv" "text" &
cpulimit -l 40 -- python3 -m utils.precomp_vecs "kaggle|lakshmi25npathi|IMDB Dataset|IMDB Dataset.csv" "review" &
cpulimit -l 40 -- python3 -m utils.precomp_vecs "github|aaroncarlfernandex|Philippine Fake News Corpus|Philippine Fake News Corpus.csv" "Headline|Content"