import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

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

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LossCompute:
    "A simple loss compute and train function."
    def __init__(self, model, generated_fertility, generated_gates,
        fetility_criterion, state_gen_criterion, gate_gen_criterion, optimizer):
        self.model = model
        self.generated_fertility = generated_fertility
        self.generated_gates = generated_gates
        self.fetility_criterion = fetility_criterion
        self.state_gen_criterion = state_gen_criterion
        self.gate_gen_criterion = gate_gen_criterion
        self.optimizer = optimizer

    def __call__(self, state_generator_out, data, is_eval=False):
        loss = 0
        fertility_out_loss = torch.Tensor([0])
        gate_out_loss = torch.Tensor([0])
        state_out_loss = torch.Tensor([0])
        state_out_nb_tokens, slot_out_nb_tokens, gate_out_nb_tokens = -1, -1, -1


        #fetility loss
        slot_out_nb_tokens = data['fertility'].view(-1).size()[0]
        fertility_out_loss = self.fetility_criterion(self.generated_fertility.view(-1, self.generated_fertility.size(-1)),
                    data['fertility'].view(-1))
        loss += fertility_out_loss

        #slot loss
        gate_out_nb_tokens = data['gates'].view(-1).size()[0]
        gate_out_loss = self.gate_gen_criterion(self.generated_gates.view(-1, self.generated_gates.size(-1)),
                    data['gates'].view(-1))
        loss += gate_out_loss

        state_out_nb_tokens = data['values_y'].view(-1).size()[0]
        state_out_loss = self.state_gen_criterion(state_generator_out.view(-1, state_generator_out.size(-1)), data['values_y'].view(-1))
        loss += state_out_loss

        if not is_eval:
            loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()

        losses = {}
        losses['fertility_loss'] = fertility_out_loss.item()
        losses['gate_loss'] = gate_out_loss.item()
        losses['state_loss'] = state_out_loss.item()

        nb_tokens = {}
        nb_tokens['slot'] = slot_out_nb_tokens
        nb_tokens['state'] = state_out_nb_tokens
        nb_tokens['gate'] = gate_out_nb_tokens

        return losses, nb_tokens
