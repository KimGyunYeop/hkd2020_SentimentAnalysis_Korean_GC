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

class REFINEEMB(nn.Module):
    def __init__(self, dic, w2v,device):
        super(REFINEEMB, self).__init__()
        self.w2v = w2v
        self.device = device
        self.dic = dic
        vectors = []
        for word in dic.keys():
            try:
                vectors.append(w2v.wv.word_vec(word))
            except:
                continue
        self.vector_parameter = nn.Parameter(torch.tensor(vectors).t(),requires_grad = True)
        self.linear = nn.Linear(len(vectors), 200, bias=False)
        self.linear.weight = self.vector_parameter
        self.softmax = nn.Softmax(dim=-1)
        self.weight = torch.FloatTensor([1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9, 1 / 10]).repeat(len(neighbors), 1).to(device)

    def distance(self, x, y):
        return torch.sum(torch.sub(x,y).mul(2),dim=-1)
    def loss(self,previous_vector,now_vector,neighbors):
        alpha=0.8
        beta = 0.2
        result1= self.softmax(self.distance(previous_vector, now_vector).mul_(alpha))
        result2 = torch.sum(self.weight.mul_(self.softmax(self.distance(now_vector.unsqueeze(1).repeat(1,10,1), neighbors))),dim=-1).mul_(beta)
        return result1 + result2

    def forward(self, previous_vector,ones_data,neighbors):
        total_loss = self.loss(previous_vector,self.linear(ones_data),neighbors).sum()
        return total_loss
#tokenizer
okt = Kkma()
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
word2vec = Word2Vec.load("ko.bin")
print(len(word2vec.wv.vocab))
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
        neighbor_sorted = sorted(neighbor_score.items(), key=lambda item: item[1],reverse=False)
        neighbor = [word2vec.wv.word_vec(k) for k, _ in neighbor_sorted]
        neighbors.append(neighbor)
    except:
        error_count+=1
        continue
print(error_count)
print(neighbor[0])
#model learning
device = "cuda:{}".format(1) if torch.cuda.is_available() else "cpu"
print(device)
model = REFINEEMB(dic_sentiment2score, word2vec,device)
model.to(device)

optimizer = AdamW(model.parameters(), lr=0.01)
for param in model.parameters():
    param.requires_grad = True

print(model)
neighbors = torch.FloatTensor(neighbors).to(device)
tmp_data = torch.ones(len(neighbors), len(neighbors), requires_grad=False).to(device)
model.train()
previous_weight=[]
for epoch in range(80):
    optimizer.zero_grad()
    previous_weight.append(model.vector_parameter.data.clone().t())
    if epoch>=1:
        loss = model(previous_weight[epoch-1],tmp_data, neighbors)
    else:
        loss = model(previous_weight[epoch], tmp_data, neighbors)
    print("loss : ", loss)
    loss.backward(create_graph=True)
    optimizer.step()
    del loss
    torch.cuda.empty_cache()
print(previous_weight[0])
print(model.linear.weight.t())
for i, (k, _) in enumerate(neighbor_sorted):
    word2vec.wv[k] = model.linear.weight.t().detach().cpu()[i].numpy()
    print(word2vec.wv[k])

word2vec.save("word2vec_refining.model")

