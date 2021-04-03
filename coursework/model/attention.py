import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class AttentionNet(nn.Module):
    "Construct attention module"
    def __init__(self, layer, N):
        super(AttentionNet, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, domainslots, context, delex_context):
        out = None
        for layer in self.layers:
            out = layer(domainslots, context, delex_context)
        return self.norm(out)


class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    "A residual connection followed by a layer norm."
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

    def expand_forward(self, x, sublayer):
        out = self.dropout(sublayer(self.norm(x)))
        out = out.mean(1).unsqueeze(1).expand_as(x)
        return x + out

    def nosum_forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x)))

class SubLayer(nn.Module):
    def __init__(self, size, attn, feedforward, dropout, nb_attn):
        super(SubLayer, self).__init__()
        self.attn = attn
        self.feedforward = feedforward
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+1)

    def forward(self, seq1, seq2, seq3):
        out = self.sublayer[0](seq1, lambda seq1: self.attn[0](seq1, seq1, seq1))
        out = self.sublayer[1](out, lambda out: self.attn[1](out, seq2, seq2))
        out = self.sublayer[2](out, lambda out: self.attn[2](out, seq3, seq3))
        return self.sublayer[3](out, self.feedforward)
