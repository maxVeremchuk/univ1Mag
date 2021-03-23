import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerGenerator(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(PointerGenerator, self).__init__()

        self.pointer_state_W = nn.Linear(d_model, vocab_size)
        self.pointer_gen_W = nn.Linear(d_model * 3, 1)

    def forward(self, domainslots, out_states, context):
        p_state_vocab = F.softmax(self.pointer_state_W(out_states), dim = -1)
        p_state_ptr = F.softmax(torch.transpose(context, 0, 1) * out_states, dim = -1)

        p_gen_vec = torch.cat([domainslots, out_states, context], -1) # .expand_as(domainslots)
        p_state_gen = nn.Sigmoid()(self.pointer_gen_W(p_gen_vec))
        p_state_out = p_state_gen * p_state_vocab + (1 - p_state_gen) * p_state_ptr
        return torch.log(p_state_out)
