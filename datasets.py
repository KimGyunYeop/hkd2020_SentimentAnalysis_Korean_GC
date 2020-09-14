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

DATASET_LIST = {
    "StarV_ANN": BaseDataset,
    "StarV_AM" : BaseDataset,
    "ENSEMBLE_MODEL" : BaseDataset
}