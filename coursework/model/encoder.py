import torch
import torch.nn as nn
import math
from attention import LayerNorm


class EncoderRNN(nn.Module):
    def __init__(self, d_model, vocab, domain_vocab, slot_vocab, nb_layers=3):
        super(EncoderRNN, self).__init__()
        layers = [Embeddings(d_model, vocab), nn.Linear(
            d_model, d_model), nn.ReLU(), PositionalEncoding(d_model)]
        self.context_embeding = nn.Sequential(*layers)

        domain_layers = [Embeddings(d_model, domain_vocab), PositionalEncoding(d_model)]
        self.domain_embeding = nn.Sequential(*domain_layers)

        slot_layers = [Embeddings(d_model, slot_vocab), PositionalEncoding(d_model)]
        self.slot_embeding = nn.Sequential(*slot_layers)

        self.norm = nn.ModuleList()
        self.nb_layers = nb_layers
        for _ in range(nb_layers):
            self.norm.append(LayerNorm(d_model))

    def forward(self, domains, slots, context, delex_context):
        context = self.context_embeding(context)
        delex_context = self.context_embeding(delex_context)
        domains = self.domain_embeding(domains)
        slots = self.slot_embeding(slots)
        domainslots = domains + slots

        return self.norm[0](domainslots), self.norm[1](context), self.norm[2](delex_context),


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
