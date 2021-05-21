import copy

from model.attention import *
from config import *
from model.encoder import *
from model.feedforward import *
from model.fertility_decoder import *
from model.pointer_generator import *
from model.state_decoder import *


class ChatBotModel(nn.Module):
    def __init__(self, fertility_decoder, state_decoder):
        super(ChatBotModel, self).__init__()
        self.fertility_decoder = fertility_decoder
        self.state_decoder = state_decoder

    def forward(self, out):
        out = self.fertility_decoder.forward(out)
        out = self.state_decoder(out)
        return out


def create_model(dictionary, domain_dicrionary, slot_dictionary, max_fert_value, args):

    cpy = copy.deepcopy

    attention_layer = MultiHeadedAttention(
        args['h_attn'], args['d_model'], dropout=args['drop'])
    feedforward_layer = PositionwiseFeedForward(
        args['d_model'], args['d_ff'], dropout=args['drop'])

    # fertility deconder init
    encoder_fert = EncoderRNN(args['d_model'], dictionary,
                              domain_dicrionary, slot_dictionary)

    nb_attn = 3

    fert_sub_layer = SubLayer(args['d_model'], cpy(attention_layer), cpy(
        feedforward_layer), args['drop'], nb_attn=nb_attn)
    fert_att_layer = AttentionNet(fert_sub_layer, args['fert_dec_N'])

    fert_generator = nn.Linear(args['d_model'], max_fert_value + 1)
    gate_generator = nn.Linear(args['d_model'], len(GATES))

    fertility_decoder = Fertility_Decoder(
        encoder_fert, fert_att_layer, fert_generator, gate_generator)

    # state deconder init
    encoder_state = cpy(encoder_fert)
    state_sub_layer = SubLayer(args['d_model'], cpy(attention_layer), cpy(
        feedforward_layer), args['drop'], nb_attn=nb_attn)
    state_att_layer = AttentionNet(state_sub_layer, args['state_dec_N'])

    pointer_attn = MultiHeadedAttention(1, args['d_model'], dropout=0.0)
    pointer_generator = PointerGenerator(len(dictionary), args['d_model'], pointer_attn)

    state_decoder = State_Decoder(encoder_state,
                                  state_att_layer, pointer_generator)

    model = ChatBotModel(fertility_decoder, state_decoder)
    return model
