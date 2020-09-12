import pickle
import os
import pandas as pd
from konlpy.tag import Okt,Kkma
from gensim.models import Word2Vec
import torch.nn as nn
import torch
from transformers import AdamW
from torch.optim import Adam
import numpy as np

#tokenizer
okt = Okt()
tkn2pol = pickle.load(open(os.path.join('../lexicon','kosac_polarity.pkl'), 'rb'))
pol2idx = ['NEG', 'None','POS']
dict_pol2idx = {y:(x-1) for x,y in enumerate(pol2idx)}
dict_pol2idx['COMP'] = 0
dict_pol2idx['NEUT'] = 0
naver_sentiment = pd.read_csv(os.path.join('../lexicon','naver_dc_all.csv'))
dic_sentiment2score = {list(okt.morphs(tokens))[0]:dict_pol2idx[pol] for tokens,pol in tkn2pol.items()}
dic_naver_sentiment2score = {list(okt.morphs(naver_sentiment["word"][i]))[0]:naver_sentiment["sentiment"][i] for i in range(len(naver_sentiment))}

dic_sentiment2score.update(dic_naver_sentiment2score)
#word2vec
word2vec = Word2Vec.load('word2vec.model')
error_count = 0
neighbors =  []
for word, score in dic_sentiment2score.items():
    try:
        neighbor = word2vec.wv.similar_by_word(word, topn=10)
        neighbor_score = {}
        for neighbor_word,_ in neighbor:
            try:
                neighbor_score[neighbor_word] = abs(dic_sentiment2score[word] - dic_sentiment2score[neighbor_word])
            except:
                neighbor_score[neighbor_word] = dic_sentiment2score[word]
                continue
        neighbor = [word2vec.wv.word_vec(k) for k, _ in sorted(neighbor_score.items(), key=lambda item: item[1],reverse=False)]
        neighbors.append(neighbor)
    except:
        error_count+=1
        continue
print(error_count)
vectors = []
for word in dic_sentiment2score.keys():
    try:
        vectors.append(word2vec.wv.word_vec(word))
    except:
        continue
weight = torch.FloatTensor([1,1/2,1/3,1/4,1/5,1/6,1/7,1/8,1/9,1/10]).repeat(len(vectors),1)
neighbors = torch.FloatTensor(neighbors)
softmax = nn.Softmax(dim=-1)
#model learning
def distance(x, y):
    return torch.sum(torch.sub(x, y).mul(2), dim=-1)

def loss( f_vectors,f_after_vectors, neighbors):
    beta = 0.5
    alpha = 1 - beta
    result = distance(f_vectors, f_after_vectors).mul_(alpha) +torch.sum(weight.mul_(distance(f_after_vectors.unsqueeze(1).repeat(1,10,1), neighbors)), dim=-1).mul_(beta)
    return result.sum()/1000
def next_vector(f_vectors):
    r = 0.5
    beta = 0.8
    vectororoo = weight.unsqueeze(2).repeat(1,1,200).view(-1,10,200).mul_(neighbors)
    result = f_vectors.mul_(r)+torch.sum(vectororoo, dim=1).mul_(beta)\
        .div(torch.sum(weight.unsqueeze(2).repeat(1,1,200).view(-1,10,200), dim=1).mul_(beta)+r)
    return result
for epoch in range(100):
    print("epoch : ",epoch)
    f_vectors = torch.tensor(vectors)
    f_after_vectors = next_vector(f_vectors)
    obj_fn =loss(f_vectors,f_after_vectors,neighbors)
    print(obj_fn)
    f_vectors = f_after_vectors

