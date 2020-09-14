import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
import numpy as np
from konlpy.tag import Twitter
from tqdm import tqdm

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

class KOSACDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        super(KOSACDataset,self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.maxlen = args.max_seq_len
        if "train" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.train_file)
        elif "dev" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.dev_file)
        elif "test" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.test_file)

        self.dataset = pd.read_csv(data_path, encoding="utf8", sep="\t")
        if "small" in mode:
            self.dataset = self.dataset[:10000]
        self.polarities, self.intensities = self.get_sentiment_data(self.dataset)

    def convert_sentiment_to_ids(self, mode, all_labels):
        pol2idx = ['None', 'POS', 'NEUT', 'COMP', 'NEG']
        int2idx = ['Medium', 'Low', 'None', 'High']
        all_ids = []
        for labels in all_labels:
            ids = []
            if mode == 'polarity':
                for label in labels:
                    ids.append(pol2idx.index(label))
            elif mode == 'intensity':
                for label in labels:
                    ids.append(int2idx.index(label))

            all_ids.append(ids)

        return all_ids

    def convert_ids_to_intensity(self, ids):
        int2idx = ['Medium', 'Low', 'None', 'High']
        dict_int2idx = {x:y for x,y in enumerate(int2idx)}
        new_ids = [dict_int2idx[x] for x in ids]

        return new_ids

    def convert_ids_to_polarity(self, ids):
        pol2idx = ['None', 'POS', 'NEUT', 'COMP', 'NEG']
        dict_pol2idx = {x:y for x,y in enumerate(pol2idx)}
        new_ids = [dict_pol2idx[x] for x in ids]

        return new_ids

    def get_sentiment_data(self, dataset):
        tkn2pol = pickle.load(open(os.path.join('lexicon','kosac_polarity.pkl'), 'rb'))
        tkn2int = pickle.load(open(os.path.join('lexicon','kosac_intensity.pkl'), 'rb'))
        polarities = []
        intensities = []

        for i in range(len(dataset)):
            tokens = self.tokenizer._tokenize(str(dataset.at[i,'review']))
            polarity = []
            intensity = []
            polarity.append('None')
            intensity.append('None')
            for token in tokens[:self.maxlen - 2]:
                if token[:2] == '##':
                    tkn = token[2:]
                else:
                    tkn = token
                try:
                    polarity.append(tkn2pol[tkn])
                    intensity.append(tkn2int[tkn])
                except KeyError:
                    polarity.append("None")
                    intensity.append("None")

            polarity.append('None')
            intensity.append('None')

            if self.maxlen - len(polarity) <= 0:
                count = 0
            else:
                count = self.maxlen - len(polarity)

            polarity = polarity + ['None' for i in range(count)]
            intensity = intensity + ['None' for i in range(count)]

            polarities.append(polarity)
            intensities.append(intensity)

        return self.convert_sentiment_to_ids('polarity', polarities), self.convert_sentiment_to_ids('intensity', intensities)

DATASET_LIST = {
    "StarV_ANN": BaseDataset,
    "StarV_AM" : BaseDataset,
    "KOSAC_LSTM_ATT": KOSACDataset,
    "VoSenti_for_Word": BaseDataset,
    "ENSEMBLE_MODEL" : BaseDataset
}