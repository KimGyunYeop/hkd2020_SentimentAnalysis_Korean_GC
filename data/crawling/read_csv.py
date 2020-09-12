import pandas as pd

data1 = pd.read_csv("../data/watcha_Reviews.txt", sep="\t", encoding="utf8")
data2 = pd.read_csv("../data/kinolights_Reviews.txt", sep="\t", encoding="utf8")
print(len(data1))
print(len(data2))
data = pd.concat([data1,data2]).reset_index(drop=True)
print(len(data[data["label"]==1]))
print(len(data[data["label"]==0]))
print(len(data))