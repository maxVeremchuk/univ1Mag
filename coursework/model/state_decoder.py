import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


class State_Decoder(nn.Module):
    def __init__(self, encoder,
                 state_attention, pointer_generator):
        super(State_Decoder, self).__init__()
        self.encoder = encoder
        self.state_attention = state_attention
        self.pointer_generator = pointer_generator

    def forward(self, out):
        domainslots = out['encoded_domainslots']
        delex_context = out['encoded_delex_context']
        context = out['encoded_context']
        out_states = self.state_attention(domainslots, context, delex_context)
        generated = self.pointer_generator(
            domainslots, out_states, context).max(dim=-1)[1]

        out['out_states'] = out_states
        out['generated_y'] = generated
        return out
