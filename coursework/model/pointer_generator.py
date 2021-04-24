import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerGenerator(nn.Module):

    def __init__(self, vocab_size, d_model, pointer_attn):
        super(PointerGenerator, self).__init__()

        self.pointer_state_W = nn.Linear(d_model, vocab_size)
        self.pointer_gen_W = nn.Linear(d_model * 3, 1)

        self.pointer_attn = pointer_attn

    def forward(self, domainslots, out_states, context, context_plain):
        self.pointer_attn(out_states, context, context)
        pointer_attn = self.pointer_attn.attn.squeeze(1)

        p_state_vocab = F.softmax(self.pointer_state_W(out_states), dim=-1)

        context_index = context_plain.unsqueeze(1).expand_as(pointer_attn)
        p_state_ptr = torch.zeros(p_state_vocab.size())
        p_state_ptr.scatter_add_(2, context_index, pointer_attn)

        expanded_pointer_attn = pointer_attn.unsqueeze(-1).repeat(1, 1, 1, context.shape[-1])
        context_vec = (context.unsqueeze(1).expand_as(expanded_pointer_attn) * expanded_pointer_attn).sum(2)

        p_gen_vec = torch.cat([out_states, context_vec, domainslots], -1)
        p_state_gen = nn.Sigmoid()(self.pointer_gen_W(p_gen_vec)).expand_as(p_state_ptr)

        p_state_out = p_state_gen * p_state_vocab + \
            (1 - p_state_gen) * p_state_ptr

        return torch.log(p_state_out)
