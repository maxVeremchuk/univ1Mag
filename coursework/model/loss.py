import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LossCompute:
    "A simple loss compute and train function."

    def __init__(self, model,
                 fetility_criterion, state_gen_criterion, gate_gen_criterion, optimizer):
        self.model = model
        self.fetility_criterion = fetility_criterion
        self.state_gen_criterion = state_gen_criterion
        self.gate_gen_criterion = gate_gen_criterion
        self.optimizer = optimizer

    def __call__(self, data, state_generator_out, generated_fertility, generated_gates, is_eval=False):
        loss = 0.0
        fertility_out_loss = torch.Tensor([0])
        gate_out_loss = torch.Tensor([0])
        state_out_loss = torch.Tensor([0])
        state_out_nb_tokens, slot_out_nb_tokens, gate_out_nb_tokens = -1, -1, -1

        # fetility loss
        slot_out_nb_tokens = data['fertility'].view(-1).size()[0]
        fertility_out_loss = self.fetility_criterion(generated_fertility.view(-1, generated_fertility.size(-1)),
                                                     data['fertility'].view(-1))
        loss += fertility_out_loss

        print("\nfertility_out_loss" + str(fertility_out_loss))

        # gate loss
        gate_out_nb_tokens = data['gates'].view(-1).size()[0]
        gate_out_loss = self.gate_gen_criterion(generated_gates.view(-1, generated_gates.size(-1)),
                                                data['gates'].view(-1))
        loss += gate_out_loss

        print("gate_out_loss" + str(gate_out_loss))

        # slot loss
        state_out_nb_tokens = data['slot_values'].view(-1).size()[0]
        state_out_loss = self.state_gen_criterion(
            state_generator_out.view(-1, state_generator_out.size(-1)), data['slot_values'].view(-1))
        loss += state_out_loss

        print("state_out_loss" + str(state_out_loss))
        if not is_eval:
            loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()

        losses = {}
        losses['fertility_loss'] = float(fertility_out_loss)
        losses['gate_loss'] =  float(gate_out_loss)
        losses['state_loss'] = float(state_out_loss)

        nb_tokens = {}
        nb_tokens['slot'] = slot_out_nb_tokens
        nb_tokens['state'] = state_out_nb_tokens
        nb_tokens['gate'] = gate_out_nb_tokens

        return losses, nb_tokens


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0, run_softmax=True):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.run_softmax = run_softmax

    def forward(self, x, target):
        if self.run_softmax:
            x = F.log_softmax(x, dim=-1)
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        if self.padding_idx != -1:
            # including padding token
            true_dist.fill_(self.smoothing / (self.size - 2))
        else:
            # no padding token
            true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        if self.padding_idx != -1:
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx)
            if (mask.sum() > 0 and len(mask) > 0):
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
