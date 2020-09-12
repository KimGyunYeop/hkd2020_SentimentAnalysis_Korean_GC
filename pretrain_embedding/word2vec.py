from konlpy.tag import Okt,Kkma
import glob
import re
from gensim.models import Word2Vec
file_list = glob.glob("../corpus/*.txt")
print(file_list)
tokenize_sentences = []
okt = Okt()
for file_name in file_list:
    print("file name : ",file_name)
    my_file = open(file_name, "r", encoding="utf16")
    sentences = my_file.readlines()
    start_index = sentences.index("<text>\n")
    for sentence in sentences[start_index:]:
        m = re.sub(r'<[a-zA-Z.0-9"= ]*>', '', sentence)
        m = re.sub(r'</[a-zA-Z.0-9"= ]*>', '', m)
        m = re.sub(r'<[a-zA-Z.0-9"= ]*/>', '', m)
        m_ = re.sub(r'<[a-zA-Z0-9."= ]*"[^0-9]*"/>', '', m)


        if 'vocal desc' in m_:
            m = re.sub('<vocal desc=', '', m_)
            m = re.sub('/>', ' ', m)
        elif 'event desc' in m_:
            m = re.sub('<event desc=', '', m_)
            idx = m.index(">")
            m = m[:idx]
        else:
            m = m_
        found_sentence = re.sub(r'[\t\n\r]', '', m_)
        if len(found_sentence.replace(" ", "")) > 0:
            tokenize_sentence = list(okt.morphs(found_sentence))
            tokenize_sentences.append(tokenize_sentence)
embedding_model = Word2Vec.load('ko.bin')
#embedding_model = Word2Vec(tokenize_sentences, size=200,workers=1)
embedding_model.train(tokenize_sentences, total_examples=len(tokenize_sentences), epochs=10)
print(len(embedding_model.wv.vocab))
embedding_model.save("word2vec.model")