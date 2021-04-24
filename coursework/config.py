UNK_token_id = 0
PAD_token_id = 1
SOS_token_id = 2
EOS_token_id = 3

UNK_token = ' UNK '
PAD_token = ' PAD '
SOS_token = ' SOS '
EOS_token = ' EOS '

train_data_path = 'data/train_dials.json'
dev_data_path = 'data/dev_dials.json'
test_data_path = 'data/test_dials.json'
ontology_path = 'data/ontology.json'

GATES = {"gen": 0, "dontcare": 1, "none": 2}

import argparse
parser = argparse.ArgumentParser(description='Non-autoregressive DST')

parser.add_argument('-batch_size', '--batch_size', help='', required=False, type=int, default=32)
parser.add_argument('-d', '--d_model', help='', required=False, type=int, default=256)
parser.add_argument('-d_ff', '--d_ff', help='', required=False, type=int, default=1024)
parser.add_argument('-h_attn', '--h_attn', help='', required=False, type=int, default=16)
parser.add_argument('-dr','--drop', help='Drop Out', required=False, type=float, default=0.1)

parser.add_argument('-fert_dec_N', '--fert_dec_N', help='', required=False, type=int, default=3)
parser.add_argument('-state_dec_N', '--state_dec_N', help='', required=False, type=int, default=3)

args = vars(parser.parse_args())
