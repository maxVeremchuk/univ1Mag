import torch
import torch.nn as nn


class Fertility_Decoder:
    def __init__(self, encoder,
                 fertility_attention, fertility_generator, gate_generator):
        super(Fertility_Decoder, self).__init__()
        self.encoder = encoder
        self.fertility_generator = fertility_generator
        self.gate_generator = gate_generator
        self.fertility_attention = fertility_attention

    def forward(self, out):
        #print(out)
        domainslots, context, delex_context = self.encoder(out['domains'], out['slots'], out['context'], out['delex_context'])

        out_slots = self.fertility_attention(domainslots, context, delex_context)
        out['encoded_context'] = context
        out['encoded_delex_context'] = delex_context
        out['encoded_domainslots'] = domainslots
        out['out_slots'] = out_slots

        generated_fertility = self.fertility_generator(out_slots)#.max(dim=-1)[1]
        generated_gates = self.gate_generator(out_slots)#.max(dim=-1)[1]
        out['generated_fertility'] = generated_fertility
        out['generated_gates'] = generated_gates
        return out
