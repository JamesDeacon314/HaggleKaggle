import pandas as pd
import numpy as np
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

df_train = pd.read_csv("data/train.txt", delimiter="\t", encoding="latin_1",
header = None)
rated_movies = df_train.iloc[:,1].unique()
train_data = load_from_df(df_train, Reader())

df_test = pd.read_csv("data/test.txt", delimiter="\t", encoding="latin_1",
header = None)
print(df_test.shape)
df_test = df[~df.iloc[:,1].isin(rated_movies)]
print(df_test.shape)
train_data = load_from_df(df_test, Reader())

algo = SVD()
algo.fit(train_data)

print()
