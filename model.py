import torch
import torch.nn.functional as F
from torch import nn

from src import (
    MODEL_ORIGINER
)


class StarV_AM(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(StarV_AM, self).__init__()
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
                        torch.ones(batch_size).to(self.config.device))

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs)

        return result


class StarV_ANN(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(StarV_ANN, self).__init__()
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


class ENSEMBLE_MODEL(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(ENSEMBLE_MODEL, self).__init__()
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

        self.dense = nn.Linear(768*2, 768)
        self.dropout = nn.Dropout(0.2, inplace=False)
        self.out_proj = nn.Linear(768, 2)
        self.star_emb = nn.Embedding(2, 768)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        #test

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
        batch_size, max_len, _ = sig_output.shape
        zeros = torch.zeros(batch_size, max_len, dtype=torch.long).to(self.config.device)
        ones = torch.ones(batch_size, max_len, dtype=torch.long).to(self.config.device)
        emb_result = self.sentiment_embedding(zeros) * sig_output[:, :, 0].unsqueeze(-1).repeat(1, 1,
                                                                                                768) + self.sentiment_embedding(
            ones) * sig_output[:, :, 1].unsqueeze(-1).repeat(1, 1, 768)
        senti_output = self.gelu(lstm_outputs + emb_result)
        return senti_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        sentiment_outputs = self.sentiment_net(outputs[0])

        sentiment_outputs, (h, _) = self.lstm(sentiment_outputs)

        # attention
        attention_outputs = self.attention_net(sentiment_outputs, input_ids)
        sentance_emb_out = outputs[0][:, 0, :]

        concat_output = torch.cat((attention_outputs, sentance_emb_out), -1)

        outputs = self.dense(concat_output)
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
                        torch.ones(batch_size).to(self.config.device))

        result = ((loss1, 0.5* loss2, 0.5* loss3), outputs)
        return result


MODEL_LIST = {
    "StarV_ANN": StarV_ANN,
    "StarV_AM" : StarV_AM,
    "ENSEMBLE_MODEL":ENSEMBLE_MODEL
}