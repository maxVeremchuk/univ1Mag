import torch
import torch.nn as nn


class Fertility_Decoder:
    def __init__(self, encoder,
                 fertility_attention, fetrility_generator, gate_generator,
                 context_embeding, domain_embeding, slot_embeding):
        super(Fertility_Decoder, self).__init__()
        self.encoder = encoder
        self.fetrility_generator = fetrility_generator
        self.gate_generator = gate_generator
        self.fertility_attention = fertility_attention
        self.context_embeding = context_embeding
        self.domain_embeding = domain_embeding
        self.slot_embeding = slot_embeding

    def forward(self, b):
        out = {}
        domainslots, context, delex_context = self.encoder(b['context'], b['delex_context'], b['domainslots'])
        out_slots = self.fertility_attention(domainslots, context, delex_context)
        out['encoded_context'] = context
        out['encoded_delex_context'] = delex_context
        out['encoded_domainslots'] = domainslots
        out['out_slots'] = out_slots

        generated_fetrility = self.fetrility_generator(out_slots)
        generated_gates = self.gate_generator(out_slots)
        return out, generated_fetrility, generated_gates
