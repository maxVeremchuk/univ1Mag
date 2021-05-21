from tqdm import tqdm
import torch.nn as nn
import os
import os.path 
import pickle as pkl 
import json 
from model.evaluator import *
from model.predict import *


def run_test(saved_data, model):
	pbar = tqdm(enumerate(saved_data['test']),total=len(saved_data['test']), desc="evaluating", ncols=0)

	predictions = {}
	latencies = []
	oracle_predictions = {}
	joint_gate_matches = 0
	joint_fertility_matches = 0
	total_samples = 0

	dictionary = saved_data['dictionary']
	domain_dicrionary = saved_data['domain_dicrionary']
	slot_dictionary = saved_data['slot_dictionary']

	for i, data in pbar:

		predictions, latencies = predict(data, model, dictionary, domain_dicrionary, slot_dictionary, predictions)
		out = model.forward(data)
		matches, oracle_predictions = get_matches(out, data, model, dictionary, domain_dicrionary, slot_dictionary, oracle_predictions)
		joint_fertility_matches += matches['joint_fertility']
		#print(joint_fertility_matches)
		joint_gate_matches += matches['joint_gate']
		total_samples += len(data['turn_id'])

	avg_latencies = sum(latencies)/len(latencies)
	with open(args['path'] + '/latency_eval.csv', 'w') as f:
		f.write(str(avg_latencies))

	joint_acc_score, F1_score, turn_acc_score = -1, -1, -1
	oracle_joint_acc, oracle_f1, oracle_acc = -1, -1, -1

	joint_acc_score, F1_score, turn_acc_score = evaluator.evaluate_metrics(predictions)
	oracle_joint_acc, oracle_f1, oracle_acc = evaluator.evaluate_metrics(oracle_predictions)

	joint_fertility_acc = 1.0 * joint_fertility_matches / (total_samples * len(saved['all_slots']))
	joint_gate_acc = 1.0 * joint_gate_matches / (total_samples * len(saved['all_slots'])) 

	with open(args['path'] + '/eval_log.csv', 'a') as f:
		f.write("{},{},{},{},{},{},{},{}".
			format(joint_gate_acc, joint_fertility_acc,
				   joint_acc_score,turn_acc_score, F1_score,
				   oracle_joint_acc,oracle_acc,oracle_f1))

	json.dump(predictions, open(args['path'] + '/predictions.json', 'w'), indent=4)
	json.dump(oracle_predictions, open(args['path'] + '/oracle_predictions.json', 'w'), indent=4)

saved = pkl.load(open(args['path'] + '/data.pkl', 'rb'))

if int(args['eval_epoch']) > -1: 
    model = torch.load(args['path'] + '/model_epoch{}.pth.tar'.format(args['eval_epoch']))
else:
    model = torch.load(args['path'] + '/model_best.pth.tar') 

evaluator = Evaluator(saved['all_slots'])

with open(args['path'] + '/eval_log.csv', 'w') as f:
    f.write('joint_gate_acc,joint_fertility_acc,joint_acc,turn_acc_score,f1,oracle_joint_acc,oracle_slot_acc,oracle_f1\n')
model.eval()

run_test(saved, model)   