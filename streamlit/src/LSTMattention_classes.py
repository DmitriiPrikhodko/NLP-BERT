import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import torchutils as tu
from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class Config:
    n_layers: int
    embedding_size: int
    hidden_size: int
    vocab_size: int
    device: str
    seq_len: int
    bidirectional: Union[bool, int] = False


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_key = nn.Linear(hidden_size, hidden_size)
        self.linear_query = nn.Linear(hidden_size, hidden_size)
        self.cls = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, lstm_outputs, final_hidden):
        # print(f"LSTM output shape: {lstm_outputs.shape}")
        # print(f"Final_hidden shape: {final_hidden.shape}")
        keys = self.linear_key(lstm_outputs)  # (batch_size, seq_len, hidden_size)
        # print(f"After linear keys shape: {keys.shape}")
        query = self.linear_query(final_hidden)  # (batch_size, hidden_size)
        query = query.unsqueeze(1)  # (batch_size, 1, hidden_size)
        # print(f"After linear query shape: {query.shape}")
        x = self.tanh(keys + query)  # (batch_size, seq_len, hidden_size)
        # print(f"After + X shape: {x.shape}")
        x = self.cls(x)  # (batch_size, seq_len, 1)
        # print(f"After cls x shape: {x.shape}")
        x = x.squeeze(-1)  # (batch_size, seq_len)
        # print(f"After squeeze x shape: {x.shape}")
        attention_weights = F.softmax(x, dim=-1)  # (batch_size, seq_len)
        # print(f"Attention weights shape: {attention_weights.shape}")
        attention_weights_bmm = attention_weights.unsqueeze(
            1
        )  # (batch_size, 1, seq_len)
        # print(f"Attention weights for bmm shape: {attention_weights_bmm.shape}")

        # bmm : (batch_size, 1, seq_len) * (batch_size, seq_len, hidden_size) = (batch_size, 1, hidden_size)
        context = torch.bmm(attention_weights_bmm, keys)  # (batch_size, 1, hidden_size)
        # print(f"Context shape: {context.shape}")
        context = context.squeeze(1)
        # print(f"Context final shape: {context.shape}")

        return context, attention_weights


class LSTMBahdanauAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # инициализируем конфиг
        self.config = config
        self.seq_len = self.config.seq_len
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size
        self.emb_size = self.config.embedding_size
        self.n_layers = self.config.n_layers
        self.device = self.config.device
        self.bidirectional = bool(self.config.bidirectional)

        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(np.zeros((self.vocab_size, self.emb_size)))
        )
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.bidirect_factor = 2 if self.bidirectional == 1 else 1
        self.attn = BahdanauAttention(self.hidden_size)
        self.clf = nn.Sequential(
            nn.Linear(self.hidden_size, 128), nn.Dropout(), nn.Tanh(), nn.Linear(128, 1)
        )

    def model_description(self):
        direction = "bidirect" if self.bidirectional else "onedirect"
        return f"rnn_{direction}_{self.n_layers}"

    def forward(self, x):
        embeddings = self.embedding(x)
        outputs, (h_n, _) = self.lstm(embeddings)
        # att_hidden, att_weights = self.attn(outputs, h_n[-1].squeeze(0))
        att_hidden, att_weights = self.attn(outputs, h_n[-1])
        out = self.clf(att_hidden)
        return out, att_weights
