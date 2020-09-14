import pandas as pd

df = pd.read_csv('./data/naverMovie_Reviews_2017.txt', sep='\t')
df = df.drop_duplicates(subset = ['reviews'])
all_df = df
pos = df[df['label']==1]
print(len(pos))
neg = df[df['label']==0]
print(len(neg))

df = pd.read_csv('./data/naverMovie_Reviews_2018.txt', sep='\t')
df = df.drop_duplicates(subset = ['reviews'])
all_df = all_df.append(df)
pos = df[df['label']==1]
print(len(pos))
neg = df[df['label']==0]
print(len(neg))
df = pd.read_csv('./data/naverMovie_Reviews_2019.txt', sep='\t')
df = df.drop_duplicates(subset = ['reviews'])
all_df = all_df.append(df)
pos = all_df[all_df['label']==1]
print(len(pos))
neg = all_df[all_df['label']==0]
print(len(neg))

pos_sample = pos.sample(n=20000)
print(len(pos_sample))
neg_sample = neg.sample(n=25000)
print(len(neg_sample))

pos_sample.to_csv('./data/naver_posSample.txt')
neg_sample.to_csv('./data/naver_negSample.txt')

