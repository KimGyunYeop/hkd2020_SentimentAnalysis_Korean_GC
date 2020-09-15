import os
import random
import logging

import torch
import numpy as np

from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    ElectraModel
)

CONFIG_CLASSES = {
    "koelectra-base": ElectraConfig
}

TOKENIZER_CLASSES = {
    "koelectra-base": ElectraTokenizer,
}

MODEL_ORIGINER = {
    "koelectra-base": ElectraModel
}



def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }


def pearson_and_spearman(labels, preds):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def f1_pre_rec(labels, preds):
    return {
        "precision": precision_score(labels, preds, suffix=True),
        "recall": recall_score(labels, preds, suffix=True),
        "f1": f1_score(labels, preds, suffix=True)
    }


def show_ner_report(labels, preds):
    return classification_report(labels, preds, suffix=True)


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return acc_score(labels, preds)