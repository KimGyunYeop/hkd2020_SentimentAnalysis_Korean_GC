import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import re
import random

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        super(BaseDataset,self).__init__()
        self.tokenizer = tokenizer
        self.maxlen = args.max_seq_len
        if "train" in mode:
            data_path = os.path.join(args.data_dir, args.train_file)
        elif "dev" in mode:
            data_path = os.path.join(args.data_dir,  args.dev_file)
        elif "test" in mode:
            data_path = os.path.join(args.data_dir, args.test_file)
        self.dataset = pd.read_csv(data_path, encoding="utf8", sep="\t")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        txt = str(self.dataset.at[idx,"review"])
        data = self.tokenizer(txt, pad_to_max_length=True, max_length=self.maxlen, truncation=True)
        input_ids = torch.LongTensor(data["input_ids"])
        token_type_ids = torch.LongTensor(data["token_type_ids"])
        attention_mask = torch.LongTensor(data["attention_mask"])
        label = self.dataset.at[idx,"rating"]

        return (input_ids, token_type_ids, attention_mask, label),txt

class AugmentBaseDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        super(AugmentBaseDataset,self).__init__()
        self.tokenizer = tokenizer
        self.maxlen = args.max_seq_len
        if "train" in mode:
            data_path = os.path.join(args.data_dir, args.train_file)
        elif "dev" in mode:
            data_path = os.path.join(args.data_dir, args.dev_file)
        elif "test" in mode:
            data_path = os.path.join(args.data_dir, args.test_file)
        self.dataset = pd.read_csv(data_path, encoding="utf8", sep="\t")
        if "small" in mode:
            self.dataset = self.dataset[:10000]

        lexicon_path = os.path.join(args.data_dir, "korean_lexicon", "AugData.txt")
        self.lexicon = pd.read_csv(lexicon_path, encoding="utf8", sep="\t")
        print(self.lexicon)
        self.lexicon = self.lexicon[self.lexicon["type"].isin(["비슷한말","상위어","하위어"])]

        self.lexicon_dic = self.get_lexicon2dic(self.lexicon)

        self.re_compile_words = re.compile(r"(" + "|".join(self.lexicon_dic.keys()) + ")")

    def get_lexicon2dic(self,lexicon):
        lexicon_dic = {}
        for index in range(len(lexicon)):
            word1 = lexicon.iloc[index]["word1"]
            word2 = lexicon.iloc[index]["word2"]
            if word1 in lexicon_dic.keys() and word2 in lexicon_dic.keys():
                total = lexicon_dic[word1] + lexicon_dic[word2]
                lexicon_dic[word1] = total
                lexicon_dic[word2] = total
            elif word1 in lexicon_dic.keys() and not word2 in lexicon_dic.keys():
                lexicon_dic[word2] = lexicon_dic[word1]
            elif word2 in lexicon_dic.keys() and not word1 in lexicon_dic.keys():
                lexicon_dic[word1] = lexicon_dic[word2]
            else:
                lexicon_dic[word1] = [word1]
                lexicon_dic[word2] = [word2]

            lexicon_dic[word1].append(word2)
            lexicon_dic[word2].append(word1)
            lexicon_dic[word1] = list(set(lexicon_dic[word1]))
            lexicon_dic[word2] = list(set(lexicon_dic[word2]))
        return lexicon_dic

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        txt = str(self.dataset.at[idx,"review"])
        print(txt)
        lexicon_words= list(set(self.re_compile_words.findall(txt)))
        for word in lexicon_words:
            txt = txt.replace(word, random.choice(self.lexicon_dic[word]))
        print(txt)
        data = self.tokenizer(txt, pad_to_max_length=True, max_length=self.maxlen, truncation=True)
        input_ids = torch.LongTensor(data["input_ids"])
        token_type_ids = torch.LongTensor(data["token_type_ids"])
        attention_mask = torch.LongTensor(data["attention_mask"])
        label = self.dataset.at[idx,"rating"]

        return (input_ids, token_type_ids, attention_mask, label),txt


DATASET_LIST = {
    "StarV_ANN": BaseDataset,
    "StarV_AM" : BaseDataset,
    "KOSAC_LSTM_ATT": BaseDataset,
    "VoSenti_for_Word": AugmentBaseDataset,
    "ENSEMBLE_MODEL" : BaseDataset
}