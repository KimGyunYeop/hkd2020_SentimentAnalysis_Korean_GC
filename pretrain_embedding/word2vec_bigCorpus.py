#-*-coding:utf-8-*-
from konlpy.tag import Okt,Kkma
import glob
import re
from gensim.models import Word2Vec
file_list = glob.glob("../big_corpus/*.txt")
print(file_list)
tokenize_sentences = []
okt = Okt()
for file_name in file_list:
    print(file_name)
    my_file = open(file_name, "r", encoding="EUCKR",errors="ignore")
    sentences = my_file.readlines()
    for sentence in sentences:
        sentence = re.sub(r'[\t\n\r]', ' ', sentence)
        tokenize_sentence = list(okt.morphs(sentence))
        tokenize_sentences.append(tokenize_sentence)

embedding_model = Word2Vec(tokenize_sentences, size=200,workers=1)
embedding_model.train(tokenize_sentences, total_examples=len(tokenize_sentences), epochs=10)
print(len(embedding_model.wv.vocab))
embedding_model.save("word2vec.model")