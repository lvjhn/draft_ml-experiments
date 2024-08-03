from setup.transforms.agmv import AGMV
import pandas as pd

X = AGMV("glove-twitter-25").fit_transform(pd.Series([
    "Hello, what is up?",
    "The quick brown fox jumped.",
    "Over the lazy dog"
]))

print(X)
print(X.shape)