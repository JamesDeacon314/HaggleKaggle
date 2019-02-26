import pandas as pd
import numpy as np
from surprise import NormalPredictor, Dataset, SVD, Reader
from surprise.model_selection import cross_validate

df_train = pd.read_csv("data/train.txt", delimiter="\t", encoding="latin_1",
header = None)
rated_movies = df_train.iloc[:,1].unique()
train_data = Dataset.load_from_df(df_train, Reader())
train_data = train_data.build_full_trainset()
df_test = pd.read_csv("data/test.txt", delimiter="\t", encoding="latin_1",
header = None)
df_test = df_test[df_test.iloc[:,1].isin(rated_movies)]
test_data = Dataset.load_from_df(df_test, Reader())

algo = SVD()
algo.fit(train_data)
predictions = algo.predict(test_data)
accuracy.rmse(predictions, verbose=True)
U = algo.pu
V = algo.qi
