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
parser.add_argument('-wu','--warmup', help='Warmup', required=False, type=int, default=12880)
parser.add_argument('-epochs','--epochs', help='Epochs', required=False, type=int, default=100)
parser.add_argument('-path','--path', help='Path', required=False, default='saved_model')
parser.add_argument('-patience','--patience', help='Patience', required=False, type=int, default=6)
parser.add_argument('-reporting', '--reporting', help='report period during training', required=False, default=100)
parser.add_argument('-ptest', '--p_test', help='', required=False, type=float, default=1)
parser.add_argument('-ptest_ft', '--p_test_fertility', help='', required=False, type=float, default=1)

parser.add_argument('-eval_epoch', '--eval_epoch', help='Epoch of evaluation test', required=False, default=-1)

parser.add_argument('-fert_dec_N', '--fert_dec_N', help='Num of fert dec attention layers', required=False, type=int, default=3)
parser.add_argument('-state_dec_N', '--state_dec_N', help='Num of state dec attention layers', required=False, type=int, default=3)

args = vars(parser.parse_args())
