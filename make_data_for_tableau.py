import numpy as np
import matplotlib.pyplot as plt
import pandas
import csv

with open("data/movies.txt","r", encoding="windows-1250") as f:
    reader = csv.reader(f, delimiter="\t")
    movies = np.array(list(reader))


data = np.loadtxt('data/data.txt').astype(int)
print(movies.shape)


for i in range(len(data)):
    data[i][0] = 0
