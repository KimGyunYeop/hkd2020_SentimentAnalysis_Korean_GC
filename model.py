import argparse
import json
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION,
    MODEL_ORIGINER,
    init_logger,
    set_seed,
    compute_metrics
)


class BASEELECTRA(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.config = config
        self.gelu = nn.GELU()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        outputs = self.dense(outputs[0][:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        # print(loss.shape)
        # print(loss)
        # print(len(outputs))
        # print(outputs.shape)

        result = (loss, outputs)

        return result


class BASEELECTRA_COS(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.config = config
        self.gelu = nn.GELU()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        x1 = embs[:, 0, :].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = embs[:, 0, :].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)

        y = labels.unsqueeze(0).repeat(batch_size, 1).type(torch.FloatTensor).cuda()
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                        x2.view(-1, w2v_dim),
                        y.view(-1))

        result = ((loss1, loss2), outputs)

        return result


class BASEELECTRA_COS_NEG(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS_NEG, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.config = config
        self.gelu = nn.GELU()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        labels_2 = labels.type(torch.FloatTensor).cuda()
        for i in range(len(labels_2)):
            labels_2[i] = labels_2[i].double() * 2 - 1
        p_idx = (labels_2 == 1).nonzero().cuda()
        n_idx = (labels_2 == -1).nonzero().cuda()

        x1 = embs[:, 0, :].squeeze()
        x1_p = x1[p_idx]
        x1_n = x1[n_idx]
        len_p = len(x1_p)
        len_n = len(x1_n)
        loss2 = 0

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        if len_p != 0 and len_n != 0:
            x1_p = x1_p.squeeze()
            x1_p = x1_p.repeat(1, len_n)
            x1_p = x1_p.view(-1, w2v_dim)
            x1_n = x1_n.squeeze().repeat(len_p, 1)

            y = -torch.ones(len_p * len_n).type(torch.FloatTensor).cuda()

            loss2 = loss_fn(x1_p.view(-1, w2v_dim),
                            x1_n.view(-1, w2v_dim),
                            y.view(-1))

        result = ((loss1, loss2), outputs)

        return result


class BASEELECTRA_COS2(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Linear(2, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        x1 = embs[:, 0, :].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = embs[:, 0, :].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)
        y = labels.unsqueeze(0).repeat(batch_size, 1).type(torch.FloatTensor).cuda()
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                        x2.view(-1, w2v_dim),
                        y.view(-1))

        star = torch.zeros(batch_size, 2).cuda()
        star[range(batch_size), labels] = 1
        star = self.star_emb(star)

        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        torch.ones(batch_size).cuda())

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result

class BASEELECTRA_COS2_EMB(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_EMB, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        x1 = embs[:, 0, :].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = embs[:, 0, :].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)
        y = labels.unsqueeze(0).repeat(batch_size, 1).type(torch.FloatTensor).cuda()
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                        x2.view(-1, w2v_dim),
                        y.view(-1))

        star = self.star_emb(labels)

        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        torch.ones(batch_size).cuda())

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result

class BASEELECTRA_COS2_POS_POS(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_POS_POS, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        x1 = embs[:, 0, :].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = embs[:, 0, :].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)
        y = labels.unsqueeze(0).repeat(batch_size, 1).type(torch.FloatTensor).to(self.config.device)
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='sum', margin=1)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                        x2.view(-1, w2v_dim),
                        y.view(-1))

        len_p = (labels==1).sum(())
        len_n = (labels==0).sum(())
        loss2 = loss2/(len_p*len_p+len_n*len_n)
        star = self.star_emb(labels)

        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        torch.ones(batch_size).to(self.config.device))

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result

class BASEELECTRA_COS2_POS(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_POS, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        x1 = embs[:, 0, :].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = embs[:, 0, :].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)
        y = labels.unsqueeze(0).repeat(batch_size, 1).type(torch.FloatTensor).to(self.config.device)
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='sum', margin=1)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                        x2.view(-1, w2v_dim),
                        y.view(-1))

        len_p = (labels==1).sum(())
        len_n = (labels==0).sum(())
        loss2 = loss2/(len_p*len_p+len_n*len_n)

        result = ((loss1, loss2), outputs)

        return result

class BASEELECTRA_COS2_ALL_ALL(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_ALL_ALL, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        x1 = embs[:, 0, :].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = embs[:, 0, :].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)
        y = labels.unsqueeze(0).repeat(batch_size, 1).type(torch.FloatTensor).to(self.config.device)
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                        x2.view(-1, w2v_dim),
                        y.view(-1))

        star_same = self.star_emb(labels)

        loss_p = loss_fn(embs[:, 0, :].squeeze(),
                        star_same,
                        torch.ones(batch_size).to(self.config.device))

        star_dif = self.star_emb(-labels+1)
        loss_n = loss_fn(embs[:, 0, :].squeeze(),
                        star_dif,
                        -torch.ones(batch_size).to(self.config.device))

        loss3 = (loss_p + loss_n)/2

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result

class BASEELECTRA_COS2_NEG_ALL(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_NEG_ALL, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        labels_2 = labels.type(torch.FloatTensor).cuda()
        for i in range(len(labels_2)):
            labels_2[i] = labels_2[i].double() * 2 - 1
        p_idx = (labels_2 == 1).nonzero().cuda()
        n_idx = (labels_2 == -1).nonzero().cuda()

        x1 = embs[:, 0, :].squeeze()
        x1_p = x1[p_idx]
        x1_n = x1[n_idx]
        len_p = len(x1_p)
        len_n = len(x1_n)
        loss2 = 0

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        if len_p != 0 and len_n != 0:
            x1_p = x1_p.squeeze()
            x1_p = x1_p.repeat(1, len_n)
            x1_p = x1_p.view(-1, w2v_dim)
            x1_n = x1_n.squeeze().repeat(len_p, 1)

            y = -torch.ones(len_p * len_n).type(torch.FloatTensor).cuda()

            loss2 = loss_fn(x1_p.view(-1, w2v_dim),
                            x1_n.view(-1, w2v_dim),
                            y.view(-1))

        star_same = self.star_emb(labels)

        loss_p = loss_fn(embs[:, 0, :].squeeze(),
                        star_same,
                        torch.ones(batch_size).to(self.config.device))

        star_dif = self.star_emb(-labels+1)
        loss_n = loss_fn(embs[:, 0, :].squeeze(),
                        star_dif,
                        -torch.ones(batch_size).to(self.config.device))

        loss3 = (loss_p + loss_n)/2

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result

class BASEELECTRA_COS2_POS_ALL(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_POS_ALL, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        x1 = embs[:, 0, :].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = embs[:, 0, :].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)
        y = labels.unsqueeze(0).repeat(batch_size, 1).type(torch.FloatTensor).to(self.config.device)
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='sum', margin=1)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                        x2.view(-1, w2v_dim),
                        y.view(-1))

        len_p = (labels == 1).sum(())
        len_n = (labels == 0).sum(())
        loss2 = loss2 / (len_p * len_p + len_n * len_n)

        star_same = self.star_emb(labels)

        loss_p = loss_fn(embs[:, 0, :].squeeze(),
                        star_same,
                        torch.ones(batch_size).to(self.config.device))

        star_dif = self.star_emb(-labels+1)
        loss_n = loss_fn(embs[:, 0, :].squeeze(),
                        star_dif,
                        -torch.ones(batch_size).to(self.config.device))

        loss3 = (loss_p + loss_n)/2

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result

class BASEELECTRA_COS2_POS_NEG(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_POS_NEG, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        x1 = embs[:, 0, :].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = embs[:, 0, :].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)
        y = labels.unsqueeze(0).repeat(batch_size, 1).type(torch.FloatTensor).to(self.config.device)
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='sum', margin=1)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                        x2.view(-1, w2v_dim),
                        y.view(-1))

        len_p = (labels == 1).sum(())
        len_n = (labels == 0).sum(())
        loss2 = loss2 / (len_p * len_p + len_n * len_n)

        star_dif = self.star_emb(-labels+1)
        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star_dif,
                        -torch.ones(batch_size).to(self.config.device))

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result

class BASEELECTRA_COS2_ALL_NEG(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_ALL_NEG, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        x1 = embs[:, 0, :].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = embs[:, 0, :].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)
        y = labels.unsqueeze(0).repeat(batch_size, 1).type(torch.FloatTensor).to(self.config.device)
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                        x2.view(-1, w2v_dim),
                        y.view(-1))

        star = self.star_emb(labels)

        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        -torch.ones(batch_size).to(self.config.device))

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result

class BASEELECTRA_COS2_NONE_POS_NEG(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_NONE_POS_NEG, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)

        star = self.star_emb(labels)

        loss2 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        torch.ones(batch_size).to(self.config.device))

        pos_star = self.star_emb(torch.ones(1,dtype=torch.long).to(self.config.device))
        neg_star = self.star_emb(torch.zeros(1,dtype=torch.long).to(self.config.device))

        loss3 = loss_fn(pos_star,
                        neg_star,
                        -torch.ones(1).to(self.config.device))

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result


class BASEELECTRA_COS2_POS_POS_NEG(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_POS_POS_NEG, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))


        x1 = embs[:, 0, :].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = embs[:, 0, :].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)
        y = labels.unsqueeze(0).repeat(batch_size, 1).type(torch.FloatTensor).to(self.config.device)
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='sum', margin=1)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                        x2.view(-1, w2v_dim),
                        y.view(-1))

        len_p = (labels == 1).sum(())
        len_n = (labels == 0).sum(())
        loss2 = loss2 / (len_p * len_p + len_n * len_n)

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        star = self.star_emb(labels)

        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        torch.ones(batch_size).to(self.config.device))

        pos_star = self.star_emb(torch.ones(1,dtype=torch.long).to(self.config.device))
        neg_star = self.star_emb(torch.zeros(1,dtype=torch.long).to(self.config.device))

        loss4 = loss_fn(pos_star,
                        neg_star,
                        -torch.ones(1).to(self.config.device))

        result = ((loss1, 0.5 * loss2, 0.5 * loss3, 0.5 * loss4), outputs)

        return result

class BASEELECTRA_COS2_LSTM(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_LSTM, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Linear(2, 768)
        self.config = config
        self.gelu = nn.GELU()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs, _ = self.lstm(embs)
        outputs = self.dense(outputs[:, -1, :])
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        x1 = outputs[:, -1, :].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = outputs[:, -1, :].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)
        y = labels.unsqueeze(0).repeat(batch_size, 1)
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                        x2.view(-1, w2v_dim),
                        y.view(-1))

        star = torch.zeros(batch_size, 2).to(self.config.device)
        star[range(batch_size), labels] = 1
        star = self.star_emb(star)

        loss3 = loss_fn(outputs[:, -1, :].squeeze(),
                        star,
                        torch.ones(batch_size).to(self.config.device))

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result


class BASEELECTRA_COS2_NEG(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_NEG, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2, inplace=False)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Linear(2, 768)
        self.gelu = nn.GELU()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(outputs[0][:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        labels_2 = labels.type(torch.FloatTensor).to(self.config.device)
        for i in range(len(labels_2)):
            labels_2[i] = labels_2[i].double() * 2 - 1
        p_idx = (labels_2 == 1).nonzero().to(self.config.device)
        n_idx = (labels_2 == -1).nonzero().to(self.config.device)

        x1 = embs[:, 0, :].squeeze()
        x1_p = x1[p_idx]
        x1_n = x1[n_idx]
        len_p = len(x1_p)
        len_n = len(x1_n)

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        if len_p != 0 and len_n != 0:
            x1_p = x1_p.squeeze()
            x1_p = x1_p.repeat(1, len_n)
            x1_p = x1_p.view(-1, w2v_dim)
            x1_n = x1_n.squeeze().repeat(len_p, 1)

            y = -torch.ones(len_p * len_n).type(torch.FloatTensor).to(self.config.device)

            loss2 = loss_fn(x1_p.view(-1, w2v_dim),
                            x1_n.view(-1, w2v_dim),
                            y.view(-1))

        star = torch.zeros(batch_size, 2).to(self.config.device)
        star[range(batch_size), labels] = 1
        star = self.star_emb(star)

        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        torch.ones(batch_size).to(self.config.device))

        if len_p == 0 or len_n == 0:
            result = ((loss1, torch.tensor(0), loss3), outputs)
        else:
            result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result

class BASEELECTRA_COS2_NEG_EMB(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_NEG_EMB, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2, inplace=False)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.gelu = nn.GELU()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(outputs[0][:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        labels_2 = labels.type(torch.FloatTensor).to(self.config.device)
        for i in range(len(labels_2)):
            labels_2[i] = labels_2[i].double() * 2 - 1
        p_idx = (labels_2 == 1).nonzero().to(self.config.device)
        n_idx = (labels_2 == -1).nonzero().to(self.config.device)

        x1 = embs[:, 0, :].squeeze()
        x1_p = x1[p_idx]
        x1_n = x1[n_idx]
        len_p = len(x1_p)
        len_n = len(x1_n)

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        if len_p != 0 and len_n != 0:
            x1_p = x1_p.squeeze()
            x1_p = x1_p.repeat(1, len_n)
            x1_p = x1_p.view(-1, w2v_dim)
            x1_n = x1_n.squeeze().repeat(len_p, 1)

            y = -torch.ones(len_p * len_n).type(torch.FloatTensor).to(self.config.device)

            loss2 = loss_fn(x1_p.view(-1, w2v_dim),
                            x1_n.view(-1, w2v_dim),
                            y.view(-1))

        star = self.star_emb(labels)

        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        torch.ones(batch_size).to(self.config.device))

        if len_p == 0 or len_n == 0:
            result = ((loss1, torch.tensor(0), loss3), outputs)
        else:
            result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result

class BASEELECTRA_COS2_STAR_NEG(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_STAR_NEG, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2, inplace=False)
        self.out_proj = nn.Linear(768, 2)
        self.gelu = nn.GELU()
        self.star_emb = nn.Linear(2, 768)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(outputs[0][:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        labels_2 = labels.type(torch.FloatTensor).to(self.config.device)
        for i in range(len(labels_2)):
            labels_2[i] = labels_2[i].double() * 2 - 1
        p_idx = (labels_2 == 1).nonzero().to(self.config.device)
        n_idx = (labels_2 == -1).nonzero().to(self.config.device)

        x1 = embs[:, 0, :].squeeze()
        x1_p = x1[p_idx]
        x1_n = x1[n_idx]
        len_p = len(x1_p)
        len_n = len(x1_n)

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        if len_p != 0 and len_n != 0:
            x1_p = x1_p.squeeze()
            x1_p = x1_p.repeat(1, len_n)
            x1_p = x1_p.view(-1, w2v_dim)
            x1_n = x1_n.squeeze().repeat(len_p, 1)

            y = -torch.ones(len_p * len_n).type(torch.FloatTensor).to(self.config.device)

            loss2 = loss_fn(x1_p.view(-1, w2v_dim),
                            x1_n.view(-1, w2v_dim),
                            y.view(-1))
        if len_p > 1:
            star_p = torch.zeros(len_p, 2).to(self.config.device)
            star_p[range(len_p), 1] = 1
            star_p = self.star_emb(star_p)
            loss3_p = loss_fn(x1[p_idx].squeeze(),
                              star_p,
                              -torch.ones(len_p).to(self.config.device))
        if len_n > 1:
            star_n = torch.zeros(len_n, 2).to(self.config.device)
            star_n[range(len_n), 0] = 1
            star_n = self.star_emb(star_n)
            loss3_n = loss_fn(x1[n_idx].squeeze(),
                              star_n,
                              -torch.ones(len_n).to(self.config.device))
        if len_p <= 1 and len_n > 1:
            if len_p == 0:
                result = ((loss1, torch.tensor(0), torch.tensor(0), 0.5 * loss3_n), outputs)
            else:
                result = ((loss1, torch.tensor(0), torch.tensor(0), 0.5 * loss3_n), outputs)
        elif len_p > 1 and len_n <= 1:
            if len_n == 0:
                result = ((loss1, torch.tensor(0), 0.5 * loss3_p, torch.tensor(0)), outputs)
            else:
                result = ((loss1, torch.tensor(0), 0.5 * loss3_p, torch.tensor(0)), outputs)
        else:
            result = ((loss1, 0.5 * loss2,
                       float(len_p) / (len_p + len_n) / 2 * loss3_p, float(len_n) / (len_p + len_n) / 2 * loss3_n),
                      outputs)

        return result

class BASEELECTRA_COS2_STAR_NEG_EMB(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS2_STAR_NEG_EMB, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2, inplace=False)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        #test

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(outputs[0][:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        labels_2 = labels.type(torch.FloatTensor).to(self.config.device)
        for i in range(len(labels_2)):
            labels_2[i] = labels_2[i].double() * 2 - 1
        p_idx = (labels_2 == 1).nonzero().to(self.config.device)
        n_idx = (labels_2 == -1).nonzero().to(self.config.device)

        x1 = embs[:, 0, :].squeeze()
        x1_p = x1[p_idx]
        x1_n = x1[n_idx]
        len_p = len(x1_p)
        len_n = len(x1_n)

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        if len_p != 0 and len_n != 0:
            x1_p = x1_p.squeeze()
            x1_p = x1_p.repeat(1, len_n)
            x1_p = x1_p.view(-1, w2v_dim)
            x1_n = x1_n.squeeze().repeat(len_p, 1)

            y = -torch.ones(len_p * len_n).type(torch.FloatTensor).to(self.config.device)

            loss2 = loss_fn(x1_p.view(-1, w2v_dim),
                            x1_n.view(-1, w2v_dim),
                            y.view(-1))
        if len_p > 1:
            star_p = torch.ones(len_p).type(torch.LongTensor).to(self.config.device)
            star_p = self.star_emb(star_p).squeeze()
            loss3_p = loss_fn(x1[p_idx].squeeze(),
                              star_p.to(self.config.device),
                              -torch.ones(len_p).to(self.config.device))
        if len_n > 1:
            star_n = torch.zeros(len_n).type(torch.LongTensor).to(self.config.device)
            star_n = self.star_emb(star_n).squeeze()
            loss3_n = loss_fn(x1[n_idx].squeeze(),
                              star_n.to(self.config.device),
                              -torch.ones(len_n).to(self.config.device))
        if len_p <= 1 and len_n > 1:
            if len_p == 0:
                result = ((loss1, torch.tensor(0), torch.tensor(0), 0.5 * loss3_n), outputs)
            else:
                result = ((loss1, torch.tensor(0), torch.tensor(0), 0.5 * loss3_n), outputs)
        elif len_p > 1 and len_n <= 1:
            if len_n == 0:
                result = ((loss1, torch.tensor(0), 0.5 * loss3_p, torch.tensor(0)), outputs)
            else:
                result = ((loss1, torch.tensor(0), 0.5 * loss3_p, torch.tensor(0)), outputs)
        else:
            result = ((loss1, 0.5 * loss2,
                       float(len_p) / (len_p + len_n) / 2 * loss3_p, float(len_n) / (len_p + len_n) / 2 * loss3_n),
                      outputs)

        return result

class LSTM(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs, (h, c) = self.lstm(outputs[0])

        outputs = self.dense(outputs[:, -1, :])
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        # print(loss.shape)
        # print(loss)
        # print(len(outputs))
        # print(outputs.shape)

        result = (loss, outputs)

        return result


class LSTM_ATT(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM_ATT, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.gelu = nn.GELU()

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1).unsqueeze(2)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.squeeze()

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print(outputs)
        # print(len(input_ids))
        # print(len(input_ids[0]))
        # print(len(outputs))
        # print(outputs[0].shape)
        outputs, (h, c) = self.lstm(outputs[0])
        # print("lstm")
        # print(len(outputs))
        # print(outputs.shape)

        attn_output = self.re_attention(outputs, h, input_ids)

        outputs = self.dense(attn_output)
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        # print(loss.shape)
        # print(loss)
        # print(len(outputs))
        # print(outputs.shape)

        result = (loss, outputs)
        return result


class LSTM_ATT_NEG(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM_ATT_NEG, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1).unsqueeze(2)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.squeeze()

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print(outputs)
        # print(len(input_ids))
        # print(len(input_ids[0]))
        # print(len(outputs))
        # print(outputs[0].shape)
        outputs, (h, c) = self.lstm(outputs[0])
        embs = outputs
        # print("lstm")
        # print(len(outputs))
        # print(outputs.shape)

        attn_output = self.re_attention(outputs, h, input_ids)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        # print(loss.shape)
        # print(loss)
        # print(len(outputs))
        # print(outputs.shape)
        labels_2 = labels.type(torch.FloatTensor).to(self.config.device)

        for i in range(len(labels_2)):
            labels_2[i] = labels_2[i].double() * 2 - 1
        p_idx = (labels_2 == 1).nonzero().to(self.config.device)
        n_idx = (labels_2 == -1).nonzero().to(self.config.device)

        x1 = embs[:, 0, :].squeeze()
        batch_size, seq_len, w2v_dim = embs.shape
        x1_p = x1[p_idx]
        x1_n = x1[n_idx]
        len_p = len(x1_p)
        len_n = len(x1_n)

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        if len_p != 0 and len_n != 0:
            x1_p = x1_p.squeeze()
            x1_p = x1_p.repeat(1, len_n)
            x1_p = x1_p.view(-1, w2v_dim)
            x1_n = x1_n.squeeze().repeat(len_p, 1)

            y = -torch.ones(len_p * len_n).type(torch.FloatTensor).to(self.config.device)

            loss2 = loss_fn(x1_p.view(-1, w2v_dim),
                            x1_n.view(-1, w2v_dim),
                            y.view(-1))

        if len_p != 0 and len_n != 0:
            result = ((loss, loss2), outputs)
        else:
            result = ((loss, torch.tensor(0)), outputs)
        return result


class LSTM_ATT_v2(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM_ATT_v2, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False, dropout=0.2)

        # attention module
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dense_1 = nn.Linear(768, 100)
        self.dense_2 = nn.Linear(100, 1)

        # full connected
        self.fc = nn.Linear(768, 300)

        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(300, 2)

    def attention_net(self, lstm_outputs):
        M = self.tanh(self.dense_1(lstm_outputs))
        wM_output = self.dense_2(M).squeeze()
        a = self.softmax(wM_output)
        c = lstm_outputs.transpose(1, 2).bmm(a.unsqueeze(-1)).squeeze()
        att_output = self.tanh(c)

        return att_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs, (h, c) = self.lstm(outputs[0])

        # attention
        attention_outputs = self.attention_net(outputs)

        fc_outputs = self.fc(attention_outputs)

        outputs = self.dropout(fc_outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)

        return result


class LSTM_ATT_DOT(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM_ATT_DOT, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.squeeze()

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs, (h, c) = self.lstm(outputs[0])
        attn_output = self.attention_net(outputs, h)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)
        return result


class LSTM_ATT2(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM_ATT2, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w1 = nn.Parameter(torch.randn(1, 768, 10))
        self.att_w2 = nn.Parameter(torch.randn(1, 10, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.squeeze()

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w1.repeat(batch_size, 1, 1))
        att = torch.bmm(torch.tanh(att),
                        self.att_w2.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs, (h, c) = self.lstm(outputs[0])
        attn_output = self.re_attention(outputs, h, input_ids)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)
        return result


class LSTM_ATT_MIX(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM_ATT_MIX, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # attention
        self.softmax = nn.Softmax(dim=-1)
        self.dense_att = nn.Linear(768, 1)

        self.config = config
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.tanh = nn.Tanh()

    def attention_net(self, emb_outputs):
        M = self.tanh(emb_outputs)
        wM_output = self.dense_att(M).squeeze()
        a = self.softmax(wM_output)
        c = emb_outputs.transpose(1, 2).bmm(a.unsqueeze(-1)).squeeze()
        att_output = self.tanh(c)

        return att_output

    def get_3gram_Att(self, emb_outputs):
        emb_3_Grams = []
        batch_size, seq_len, w2v_dim = emb_outputs.shape
        emb_outputs_padding = torch.nn.functional.pad(emb_outputs, (0, 0, 1, 1))
        alpha = 0.5
        for i in range(1, 51):
            # emb_3_Gram = torch.mean(emb_outputs[:,i-1:i+2,:], dim=1)
            emb_3_Gram = emb_outputs_padding[:, i, :] * alpha + emb_outputs_padding[:, i + 1, :] * (
                    1 - alpha) / 2 + emb_outputs_padding[:, i - 1, :] * (1 - alpha) / 2
            emb_3_Grams.append(emb_3_Gram)

        inputs = torch.cat(emb_3_Grams, dim=-1)
        inputs = torch.reshape(inputs, (batch_size, seq_len, w2v_dim))
        output = self.attention_net(inputs)

        return output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hie_attn_output = self.get_3gram_Att(outputs[0])

        outputs = self.dense(hie_attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        result = (loss, outputs)

        return result


class LSTM_ATT_MIX_NEG(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM_ATT_MIX_NEG, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # attention
        self.softmax = nn.Softmax(dim=-1)
        self.dense_att = nn.Linear(768, 1)

        self.config = config
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.tanh = nn.Tanh()

    def attention_net(self, emb_outputs):
        M = self.tanh(emb_outputs)
        wM_output = self.dense_att(M).squeeze()
        a = self.softmax(wM_output)
        c = emb_outputs.transpose(1, 2).bmm(a.unsqueeze(-1)).squeeze()
        att_output = self.tanh(c)

        return att_output

    def get_3gram_Att(self, emb_outputs):
        emb_3_Grams = []
        batch_size, seq_len, w2v_dim = emb_outputs.shape
        emb_outputs_padding = torch.nn.functional.pad(emb_outputs, (0, 0, 1, 1))
        alpha = 0.5
        for i in range(1, 51):
            # emb_3_Gram = torch.mean(emb_outputs[:,i-1:i+2,:], dim=1)
            emb_3_Gram = emb_outputs_padding[:, i, :] * alpha + emb_outputs_padding[:, i + 1, :] * (
                    1 - alpha) / 2 + emb_outputs_padding[:, i - 1, :] * (1 - alpha) / 2
            emb_3_Grams.append(emb_3_Gram)

        inputs = torch.cat(emb_3_Grams, dim=-1)
        inputs = torch.reshape(inputs, (batch_size, seq_len, w2v_dim))
        output = self.attention_net(inputs)

        return output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        hie_attn_output = self.get_3gram_Att(outputs[0])

        outputs = self.dense(hie_attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        labels_2 = labels.type(torch.FloatTensor).to(self.config.device)

        for i in range(len(labels_2)):
            labels_2[i] = labels_2[i].double() * 2 - 1
        p_idx = (labels_2 == 1).nonzero().to(self.config.device)
        n_idx = (labels_2 == -1).nonzero().to(self.config.device)

        x1 = embs[:, 0, :].squeeze()
        batch_size, seq_len, w2v_dim = embs.shape
        x1_p = x1[p_idx]
        x1_n = x1[n_idx]
        len_p = len(x1_p)
        len_n = len(x1_n)

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        if len_p != 0 and len_n != 0:
            x1_p = x1_p.squeeze()
            x1_p = x1_p.repeat(1, len_n)
            x1_p = x1_p.view(-1, w2v_dim)
            x1_n = x1_n.squeeze().repeat(len_p, 1)

            y = -torch.ones(len_p * len_n).type(torch.FloatTensor).to(self.config.device)

            loss2 = loss_fn(x1_p.view(-1, w2v_dim),
                            x1_n.view(-1, w2v_dim),
                            y.view(-1))

        if len_p != 0 and len_n != 0:
            result = ((loss, loss2), outputs)
        else:
            result = ((loss, torch.tensor(0)), outputs)

        return result


# --KOSAC--

class KOSAC_LSTM(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KOSAC_LSTM, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, labels, token_type_ids, polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           inputs_embeds=embedding_result)
        outputs = outputs[0] + polarity_emb_result / 100 + intensity_emb_result / 100
        outputs, _ = self.lstm(outputs)

        outputs = self.dense(outputs[:, -1, :])
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)

        return result


class KOSAC_LSTM_ATT(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KOSAC_LSTM_ATT, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.gelu = nn.GELU()

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.squeeze()

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids, polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result + polarity_emb_result / 100 + intensity_emb_result / 100

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           inputs_embeds=embedding_result)
        outputs, (h, c) = self.lstm(outputs[0])

        attn_output = self.re_attention(outputs, h, input_ids)

        outputs = self.dense(attn_output)
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)
        return result


class KOSAC_LSTM_ATT_v2(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KOSAC_LSTM_ATT_v2, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False, dropout=0.2)

        # attention module
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dense_1 = nn.Linear(768, 100)
        self.dense_2 = nn.Linear(100, 1)

        # full connected
        self.fc = nn.Linear(768, 300)

        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(300, 2)

    def attention_net(self, lstm_outputs):
        M = self.tanh(self.dense_1(lstm_outputs))
        wM_output = self.dense_2(M).squeeze()
        a = self.softmax(wM_output)
        c = lstm_outputs.transpose(1, 2).bmm(a.unsqueeze(-1)).squeeze()
        att_output = self.tanh(c)

        return att_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids, polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           inputs_embeds=embedding_result)
        outputs = outputs[0] + polarity_emb_result / 100 + intensity_emb_result / 100
        outputs, (h, c) = self.lstm(outputs)

        # attention
        attention_outputs = self.attention_net(outputs)

        fc_outputs = self.fc(attention_outputs)

        outputs = self.dropout(fc_outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)

        return result


class KOSAC_LSTM_ATT_DOT(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KOSAC_LSTM_ATT_DOT, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(4, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def forward(self, input_ids, attention_mask, labels, token_type_ids, polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result + polarity_emb_result / 100 + intensity_emb_result / 100

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           inputs_embeds=embedding_result)
        outputs, (h, c) = self.lstm(outputs[0])
        attn_output = self.attention_net(outputs, h)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)
        return result


# --KNU--
class KNU_BASE(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KNU_BASE, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(4, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, labels, token_type_ids, polarity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)

        embedding_result = input_emb_result + polarity_emb_result / 100

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           inputs_embeds=embedding_result)

        outputs = self.dense(outputs[0][:, 0, :])
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)

        return result


class KNU_LSTM(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KNU_LSTM, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 300)

        self.lstm = nn.LSTM(1068, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, labels, token_type_ids, polarity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)

        embedding_result = input_emb_result

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           inputs_embeds=embedding_result)
        outputs = torch.cat([outputs[0], polarity_emb_result / 100], dim=-1)
        outputs, _ = self.lstm(outputs)

        outputs = self.dense(outputs[:, -1, :])
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)

        return result


class KNU_LSTM_ATT(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KNU_LSTM_ATT, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.squeeze()

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids, polarity_ids, intensity_ids):
        print(attention_mask.shape)
        print(labels.shape)
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           inputs_embeds=embedding_result)
        outputs = outputs[0] + polarity_emb_result / 100 + intensity_emb_result / 100
        outputs, (h, c) = self.lstm(outputs)

        attn_output = self.re_attention(outputs, h, input_ids)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)
        return result


class KNU_LSTM_ATT_v2(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KNU_LSTM_ATT_v2, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False, dropout=0.2)

        # attention module
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dense_1 = nn.Linear(768, 100)
        self.dense_2 = nn.Linear(100, 1)

        # full connected
        self.fc = nn.Linear(768, 300)

        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(300, 2)

    def attention_net(self, lstm_outputs):
        M = self.tanh(self.dense_1(lstm_outputs))
        wM_output = self.dense_2(M).squeeze()
        a = self.softmax(wM_output)
        c = lstm_outputs.transpose(1, 2).bmm(a.unsqueeze(-1)).squeeze()
        att_output = self.tanh(c)

        return att_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids, polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           inputs_embeds=embedding_result)
        outputs = outputs[0] + polarity_emb_result / 100 + intensity_emb_result / 100
        outputs, (h, c) = self.lstm(outputs)

        # attention
        attention_outputs = self.attention_net(outputs)

        fc_outputs = self.fc(attention_outputs)

        outputs = self.dropout(fc_outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)

        return result


class KNU_LSTM_ATT_DOT(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KNU_LSTM_ATT_DOT, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def forward(self, input_ids, attention_mask, labels, token_type_ids, polarity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)

        embedding_result = input_emb_result + polarity_emb_result / 100
        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           inputs_embeds=embedding_result)
        outputs, (h, c) = self.lstm(outputs[0])
        attn_output = self.attention_net(outputs, h)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)
        return result


class KNU_LSTM_ATT_DOT_ML(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KNU_LSTM_ATT_DOT_ML, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state, soft_attn_weights

    def forward(self, input_ids, attention_mask, labels, token_type_ids, polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result + polarity_emb_result / 100 + intensity_emb_result / 100

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           inputs_embeds=embedding_result)
        outputs, (h, c) = self.lstm(outputs[0])
        attn_output, soft_attn_weights = self.attention_net(outputs, h)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)
        att_label = F.softmax((torch.abs(polarity_ids) + torch.abs(intensity_ids)).float(), dim=-1)
        loss_fct = nn.CrossEntropyLoss()
        loss_att = nn.MSELoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1)) + loss_att(soft_attn_weights.squeeze(),
                                                                         att_label.long()).float()
        result = (loss, outputs)
        return result


class EMB2_LSTM(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(EMB2_LSTM, self).__init__()
        self.posemb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.negemb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        posoutputs = self.posemb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        posoutputs, (h, c) = self.lstm(posoutputs[0])

        posoutputs = self.dense(posoutputs[:, -1, :])
        posoutputs = self.dropout(posoutputs)
        posoutputs = self.out_proj(posoutputs)

        negoutputs = self.negemb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        negoutputs, (h, c) = self.lstm(negoutputs[0])

        negoutputs = self.dense(negoutputs[:, -1, :])
        negoutputs = self.dropout(negoutputs)
        negoutputs = self.out_proj(negoutputs)

        output = torch.cat((negoutputs, posoutputs), dim=1)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(output.view(-1, 2), labels.view(-1))
        # print(loss.shape)
        # print(loss)
        # print(len(outputs))
        # print(outputs.shape)

        result = (loss, output)

        return result


class EMB1_LSTM2(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(EMB1_LSTM2, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.poslstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.posdense = nn.Linear(768, 768)
        self.posout_proj = nn.Linear(768, 1)

        self.dropout = nn.Dropout(0.2)

        self.neglstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.negdense = nn.Linear(768, 768)
        self.negout_proj = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        posoutputs, (h, c) = self.poslstm(outputs[0])

        posoutputs = self.posdense(posoutputs[:, -1, :])
        posoutputs = self.dropout(posoutputs)
        posoutputs = self.posout_proj(posoutputs)

        negoutputs, (h, c) = self.neglstm(outputs[0])

        negoutputs = self.negdense(negoutputs[:, -1, :])
        negoutputs = self.dropout(negoutputs)
        negoutputs = self.negout_proj(negoutputs)

        output = torch.cat((negoutputs, posoutputs), dim=1)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(output.view(-1, 2), labels.view(-1))
        # print(loss.shape)
        # print(loss)
        # print(len(outputs))
        # print(outputs.shape)

        result = (loss, output)

        return result


class EMB_ATT_LSTM_ATT_ver2(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(EMB_ATT_LSTM_ATT_ver2, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.config = config
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)

        #sentiment module
        self.word_dense = nn.Linear(768, 2)
        self.sentiment_embedding = nn.Embedding(2, 768)
        self.softmax = nn.Softmax(dim=-1)
        # attention module
        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.gelu = nn.GELU()

    def attention_net(self, lstm_output, input):
        batch_size, seq_len = input.shape

        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def sentiment_net(self, lstm_outputs):
        result = self.word_dense(lstm_outputs)
        sig_output = self.softmax(result)
        batch_size, max_len, _=sig_output.shape
        zeros = torch.zeros(batch_size, max_len, dtype=torch.long).to(self.config.device)
        ones = torch.ones(batch_size, max_len, dtype=torch.long).to(self.config.device)
        emb_result = self.sentiment_embedding(zeros) * sig_output[:,:,0].unsqueeze(-1).repeat(1,1,768) + self.sentiment_embedding(ones) * sig_output[:,:,1].unsqueeze(-1).repeat(1,1,768)
        senti_output = self.gelu(lstm_outputs + emb_result)
        return senti_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # embedding
        emb_output = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sentiment_outputs = self.sentiment_net(emb_output[0])

        outputs, (h,_) = self.lstm(sentiment_outputs)

        # attention
        attention_outputs = self.attention_net(outputs,input_ids)

        outputs = self.dense(attention_outputs)
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)

        return result

class LSTM_ATT_NEG(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM_ATT_NEG, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.config = config
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False, dropout=0.2)

        #sentiment module
        self.word_dense = nn.Linear(768, 2)
        self.sentiment_embedding = nn.Embedding(2, 768)
        self.softmax = nn.Softmax(dim=-1)

        # attention module
        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.gelu = nn.GELU()
        self.star_emb = nn.Linear(2, 768)

    def attention_net(self, lstm_output, input):
        batch_size, seq_len = input.shape

        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # embedding
        emb_output = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        outputs, (h,_) = self.lstm(emb_output[0])
        batch_size, seq_len, w2v_dim = outputs.shape
        # attention
        attention_outputs = self.attention_net(outputs,input_ids)

        outputs = self.dense(attention_outputs)
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        labels_2 = labels.type(torch.FloatTensor).to(self.config.device)
        for i in range(len(labels_2)):
            labels_2[i] = labels_2[i].double() * 2 - 1
        p_idx = (labels_2 == 1).nonzero().to(self.config.device)
        n_idx = (labels_2 == -1).nonzero().to(self.config.device)

        x1 = emb_output[0][:, 0, :].squeeze()
        x1_p = x1[p_idx]
        x1_n = x1[n_idx]
        len_p = len(x1_p)
        len_n = len(x1_n)

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-0.5)
        if len_p != 0 and len_n != 0:
            x1_p = x1_p.squeeze()
            x1_p = x1_p.repeat(1, len_n)
            x1_p = x1_p.view(-1, w2v_dim)
            x1_n = x1_n.squeeze().repeat(len_p, 1)

            y = -torch.ones(len_p * len_n).type(torch.FloatTensor).to(self.config.device)

            loss2 = loss_fn(x1_p.view(-1, w2v_dim),
                            x1_n.view(-1, w2v_dim),
                            y.view(-1))
        if len_p > 1:
            star_p = torch.zeros(len_p, 2).to(self.config.device)
            star_p[range(len_p), 1] = 1
            star_p = self.star_emb(star_p)
            loss3_p = loss_fn(x1[p_idx].squeeze(),
                              star_p,
                              -torch.ones(len_p).to(self.config.device))
        if len_n > 1:
            star_n = torch.zeros(len_n, 2).to(self.config.device)
            star_n[range(len_n), 0] = 1
            star_n = self.star_emb(star_n)
            loss3_n = loss_fn(x1[n_idx].squeeze(),
                              star_n,
                              -torch.ones(len_n).to(self.config.device))
        if len_p <= 1 and len_n > 1:
            if len_p == 0:
                result = ((loss1, torch.tensor(0), torch.tensor(0), 0.5 * loss3_n), outputs)
            else:
                result = ((loss1, torch.tensor(0), torch.tensor(0), 0.5 * loss3_n), outputs)
        elif len_p > 1 and len_n <= 1:
            if len_n == 0:
                result = ((loss1, torch.tensor(0), 0.5 * loss3_p, torch.tensor(0)), outputs)
            else:
                result = ((loss1, torch.tensor(0), 0.5 * loss3_p, torch.tensor(0)), outputs)
        else:
            result = ((loss1, 0.5 * loss2,
                       float(len_p) / (len_p + len_n) / 2 * loss3_p, float(len_n) / (len_p + len_n) / 2 * loss3_n),
                      outputs)

        return result

class EMB_ATT_LSTM_ATT_ver2_NEG(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(EMB_ATT_LSTM_ATT_ver2_NEG, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.config = config
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)

        # sentiment module
        self.word_dense = nn.Linear(768, 2)
        self.sentiment_embedding = nn.Embedding(2, 768)
        self.softmax = nn.Softmax(dim=-1)

        # attention module
        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.gelu = nn.GELU()
        self.star_emb = nn.Linear(2, 768)

    def attention_net(self, lstm_output, input):
        batch_size, seq_len = input.shape

        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # embedding
        emb_output = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        outputs, (h, _) = self.lstm(emb_output[0])
        batch_size, seq_len, w2v_dim = outputs.shape
        # attention
        attention_outputs = self.attention_net(outputs, input_ids)

        outputs = self.dense(attention_outputs)
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        labels_2 = labels.type(torch.FloatTensor).to(self.config.device)
        for i in range(len(labels_2)):
            labels_2[i] = labels_2[i].double() * 2 - 1
        p_idx = (labels_2 == 1).nonzero().to(self.config.device)
        n_idx = (labels_2 == -1).nonzero().to(self.config.device)

        x1 = emb_output[:, 0, :].squeeze()
        x1_p = x1[p_idx]
        x1_n = x1[n_idx]
        len_p = len(x1_p)
        len_n = len(x1_n)

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=1)
        if len_p != 0 and len_n != 0:
            x1_p = x1_p.squeeze()
            x1_p = x1_p.repeat(1, len_n)
            x1_p = x1_p.view(-1, w2v_dim)
            x1_n = x1_n.squeeze().repeat(len_p, 1)

            y = -torch.ones(len_p * len_n).type(torch.FloatTensor).to(self.config.device)

            loss2 = loss_fn(x1_p.view(-1, w2v_dim),
                            x1_n.view(-1, w2v_dim),
                            y.view(-1))
        if len_p > 1:
            star_p = torch.zeros(len_p, 2).to(self.config.device)
            star_p[range(len_p), 1] = 1
            star_p = self.star_emb(star_p)
            loss3_p = loss_fn(x1[p_idx].squeeze(),
                              star_p,
                              -torch.ones(len_p).to(self.config.device))
        if len_n > 1:
            star_n = torch.zeros(len_n, 2).to(self.config.device)
            star_n[range(len_n), 0] = 1
            star_n = self.star_emb(star_n)
            loss3_n = loss_fn(x1[n_idx].squeeze(),
                              star_n,
                              -torch.ones(len_n).to(self.config.device))
        if len_p <= 1 and len_n > 1:
            if len_p == 0:
                result = ((loss1, torch.tensor(0), torch.tensor(0), 0.5 * loss3_n), outputs)
            else:
                result = ((loss1, torch.tensor(0), torch.tensor(0), 0.5 * loss3_n), outputs)
        elif len_p > 1 and len_n <= 1:
            if len_n == 0:
                result = ((loss1, torch.tensor(0), 0.5 * loss3_p, torch.tensor(0)), outputs)
            else:
                result = ((loss1, torch.tensor(0), 0.5 * loss3_p, torch.tensor(0)), outputs)
        else:
            result = ((loss1, 0.5 * loss2,
                       float(len_p) / (len_p + len_n) / 2 * loss3_p, float(len_n) / (len_p + len_n) / 2 * loss3_n),
                      outputs)

        return result


MODEL_LIST = {
    "BASEELECTRA": BASEELECTRA,
    "BASEELECTRA_COS": BASEELECTRA_COS,
    "BASEELECTRA_COS_NEG": BASEELECTRA_COS_NEG,
    "BASEELECTRA_COS2": BASEELECTRA_COS2,
    "BASEELECTRA_COS2_LSTM": BASEELECTRA_COS2_LSTM,
    "BASEELECTRA_COS2_NEG": BASEELECTRA_COS2_NEG,
    "BASEELECTRA_COS2_STAR_NEG": BASEELECTRA_COS2_STAR_NEG,
    "BASEELECTRA_COS2_STAR_NEG_EMB": BASEELECTRA_COS2_STAR_NEG_EMB,
    "BASEELECTRA_COS2_EMB" : BASEELECTRA_COS2_EMB,
    "BASEELECTRA_COS2_ALL_NEG" : BASEELECTRA_COS2_ALL_NEG,
    "BASEELECTRA_COS2_POS_POS" : BASEELECTRA_COS2_POS_POS,
    "BASEELECTRA_COS2_ALL_ALL" : BASEELECTRA_COS2_ALL_ALL,
    "BASEELECTRA_COS2_POS" : BASEELECTRA_COS2_POS,
    "BASEELECTRA_COS2_NONE_POS_NEG" : BASEELECTRA_COS2_NONE_POS_NEG,
    "BASEELECTRA_COS2_POS_POS_NEG" : BASEELECTRA_COS2_POS_POS_NEG,
    "BASEELECTRA_COS2_NEG_ALL" : BASEELECTRA_COS2_NEG_ALL,
    "BASEELECTRA_COS2_POS_ALL" : BASEELECTRA_COS2_POS_ALL,
    "BASEELECTRA_COS2_POS_NEG" : BASEELECTRA_COS2_POS_NEG,



    "LSTM": LSTM,
    "LSTM_ATT": LSTM_ATT,
    "LSTM_ATT_NEG": LSTM_ATT_NEG,
    "LSTM_ATT_v2": LSTM_ATT_v2,
    "LSTM_ATT_DOT": LSTM_ATT_DOT,
    "LSTM_ATT2": LSTM_ATT2,
    "LSTM_ATT_MIX": LSTM_ATT_MIX,
    "LSTM_ATT_MIX_NEG": LSTM_ATT_MIX_NEG,

    "EMB2_LSTM": EMB2_LSTM,
    "EMB1_LSTM2": EMB1_LSTM2,
    "LSTM_ATT_NEG": LSTM_ATT_NEG,

    "EMB_ATT_LSTM_ATT_ver2": EMB_ATT_LSTM_ATT_ver2,
    "EMB_ATT_LSTM_ATT_ver2_NEG": EMB_ATT_LSTM_ATT_ver2_NEG,
    "EMB_CLS_LSTM_ATT": EMB_CLS_LSTM_ATT,
    "BASEELECTRA_COS2_NEG_EMB":BASEELECTRA_COS2_NEG_EMB
}