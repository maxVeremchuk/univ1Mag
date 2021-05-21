import numpy as np
import torch
from config import *
import time
import copy


def predict(data, model, dictionary, domain_dicrionary, slot_dictionary, curr_prediction):
    p = args['p_test']  # probability of using the non-ground truth delex context
    ft_p = args['p_test_fertility'] # simulate probability of using the non-ground truth fertility

    latency = []
    sorted_index = np.argsort(data['turn_id'])
    #print(data)
    for i in sorted_index:
        start = time.time()

        curr_item_info = {}
        for k, v in data.items():
            curr_item_info[k] = v[i]  # i - batch position in value list
        curr_item_info['dialogue_id'] = [curr_item_info['dialogue_id']]
        curr_item_info['turn_id'] = [curr_item_info['turn_id']]
        curr_item_info['turn_belief'] = [curr_item_info['turn_belief']]
        curr_item_info['domains'] = curr_item_info['domains'].unsqueeze(0)
        curr_item_info['slots'] = curr_item_info['slots'].unsqueeze(0)
        curr_item_info['context'] = curr_item_info['context'].unsqueeze(0)
        curr_item_info['context_plain'] = [curr_item_info['context_plain']]
        curr_item_info['delex_context'] = curr_item_info['delex_context'].unsqueeze(
            0)

        if np.random.uniform() < p:
            curr_item_info['delex_context'] = get_delex_from_prediction(curr_item_info, curr_prediction, dictionary)
            #print(curr_item_info['delex_context'])

        out = model.fertility_decoder.forward(
            curr_item_info)  # first encoder decoder

        # print(out['generated_gates'])
        generated_fertility = out['generated_fertility'].squeeze(0).max(dim=-1)[1]
        generated_gates = out['generated_gates'].squeeze(0).max(dim=-1)[1]
        
        dontcare_out = []
        slots_out = []
        domains_out = []

        if np.random.uniform() < ft_p:
            slots_out, domains_out, dontcare_out = get_state_encoder_inport_from_generated_fertility(
                generated_fertility, generated_gates, curr_item_info["slots"].squeeze(0), curr_item_info["domains"].squeeze(0))

            if len(slots_out) == 0:
                print("len slots out is 0---------------")
                print(slots_out)
                print(domains_out)
                print(dontcare_out)
                print("-"*20)

            slots_out = torch.stack(slots_out).long().unsqueeze(0)
            domains_out = torch.stack(domains_out).long().unsqueeze(0)

            curr_item_info["slots_fertility"] = slots_out
            curr_item_info["domain_fertility"] = domains_out
        out = model.state_decoder(curr_item_info)  # second decoder

        generated_states = out['generated_y'].max(dim=-1)[1]

        curr_prediction = generate_prediction(curr_item_info, generated_fertility, generated_gates, slots_out, domains_out,
                                              dontcare_out, generated_states, dictionary, domain_dicrionary, slot_dictionary, curr_prediction)
        end = time.time()
        elapsed_time = end - start
        latency.append(elapsed_time)

    return curr_prediction, latency


def get_matches(out, data, model, dictionary, domain_dicrionary, slot_dictionary, curr_prediction):
    joint_fertility_acc, joint_gates_acc = 0, 0

    _, fertility_idxs = out['generated_fertility'].max(dim=-1)
    fertility_compared = (fertility_idxs == data['fertility'])
    joint_fertility_acc = fertility_compared.long().sum().item()
    #joint_fertility_acc = ((fertility_compared != 1).sum(-1) == 0).sum().item()

    

    _, gates_idxs = out['generated_gates'].max(dim=-1)
    gates_compared = (gates_idxs == data['gates'])
    joint_gates_acc = gates_compared.long().sum().item()
    #joint_gates_acc = ((gates_compared != 1).sum(-1) == 0).sum().item()


    _, states_idxs = out['generated_y'].max(dim=-1)

    # slots_out, domains_out, dontcare_out = get_state_encoder_inport_from_generated_fertility(
    #     fertility_idxs, gates_idxs, data["slots"].squeeze(0), data["domains"].squeeze(0))
    #print( out['generated_y'].size())
    curr_prediction = generate_prediction(data, out['generated_fertility'], out['generated_gates'], data['slots_fertility'], data['domain_fertility'],
                                          [], states_idxs, dictionary, domain_dicrionary, slot_dictionary, curr_prediction)

    matches = {}
    matches['joint_fertility'] = joint_fertility_acc
    matches['joint_gate'] = joint_gates_acc
    return matches, curr_prediction


def get_delex_from_prediction(data, predictions, dictionary):
    dialogue_id = data['dialogue_id'][0]
    turn_id = data['turn_id'][0]
    delex_context_plain = data['delex_context_plain']
    if dialogue_id not in predictions or turn_id-1 not in predictions[dialogue_id]:
        return data['delex_context']
    prev_bs = predictions[dialogue_id][turn_id-1]['predicted_belief']
    context = data['context_plain'][0].split()
    delex_context = copy.copy(context)
    delex_context_plain = delex_context_plain.split()
    sys_sos_index = [idx for idx,t in enumerate(delex_context) if t == 'SOS'][1::2]
    user_sos_index = [idx for idx,t in enumerate(delex_context) if t == 'SOS'][::2]
    
    for bs in prev_bs:
        bs_tokens = bs.split('-')
        d, s, v = bs_tokens[0], bs_tokens[1], '-'.join(bs_tokens[2:])
        ds = '-'.join([d,s])

        v_tokens = v.split()
        temp = user_sos_index[:-1]
        for idx, u_idx in enumerate(temp):
            s_idx = sys_sos_index[idx]
            for t_idx, token in enumerate(delex_context[u_idx:s_idx]):
                pos = t_idx + u_idx
                if len(delex_context[pos].split('-')) == 2: continue
                if token in v_tokens:
                    delex_context[pos] = ds
        temp = user_sos_index[1:]
        for idx, u_idx in enumerate(temp):
            s_idx = sys_sos_index[idx]
            for t_idx, token in enumerate(delex_context[s_idx:u_idx]):
                pos = t_idx + s_idx
                delex_context[pos] = delex_context_plain[pos]

    for idx, token in enumerate(delex_context[user_sos_index[-1]:]):
        pos = idx + user_sos_index[-1]
        delex_context[pos] = context[pos]

    #print(delex_context)
    out = []
    for token in delex_context:
        token_index = dictionary.word2index[token] if token in dictionary.word2index else dictionary.word2index['UNK']
        out.append(token_index)
    out = torch.Tensor(out).unsqueeze(0).long()

    return out

def get_state_encoder_inport_from_generated_fertility(fertilities, gates, slots, domains):
    dict_value_out = {}
    dontcare_out = []

    if len(fertilities) == 0:
        return [], [], []

    for i in range(fertilities.shape[0]):
        fertility = fertilities[i]
        gate = gates[i]
        slot = slots[i]
        domain = domains[i]

        if gate == GATES['none']:
            continue
        elif gate == GATES['dontcare']:
            dontcare_out.append((slot.long().item(), domain.long().item()))
            continue

        if slot in [0, 1, 2, 3]:  # UNK, PAD, SOS, EOS
            continue

        if fertility == 0:  # 0 words to write slot
            continue

        if (slot, domain) not in dict_value_out:
            dict_value_out[(slot, domain)] = 0

        dict_value_out[(slot, domain)] = max(
            [fertility, dict_value_out[(slot, domain)]])

        slots_out = []
        domains_out = []
        for slot_domain, fertility in dict_value_out.items():
            for _ in range(fertility):
                slots_out.append(slot_domain[0])
                domains_out.append(slot_domain[1])

    return slots_out, domains_out, dontcare_out

#TODO:remove redundant params
def generate_prediction(data, generated_fertility, generated_gates, slots_out, domains_out,
                        dontcare_out, generated_states, dictionary, domain_dicrionary, slot_dictionary, curr_prediction):
    
    sorted_index = np.argsort(data['turn_id'])
    
    for idx in sorted_index:
        dialogue_id = data['dialogue_id'][idx]
        turn_id = data['turn_id'][idx]
        belief_state = data['turn_belief'][idx]

        states = [dictionary.index2word[i]
                  for i in generated_states[idx].tolist()]
        # print(slots_out[idx])
        slots = [slot_dictionary.index2word[i]
                 for i in slots_out[idx].tolist()]
        domains = [domain_dicrionary.index2word[i]
                   for i in domains_out[idx].tolist()]

        state_gathered = {}
        for idx in range(len(domains)):
            state = states[idx]
            slot = slots[idx]
            domain = domains[idx]
            if 'PAD' in [domain, slot, state]:
                continue
            if 'EOS' in [domain, slot, state]:
                continue
            if 'SOS' in [domain, slot, state]:
                continue
            if 'UNK' in [domain, slot, state]:
                continue
            if 'dontcare' in [state]:
                continue
            if 'none' in [state]:
                continue

            slot = slot.replace("_SLOT", "")
            domain = domain.replace("_DOMAIN", "")
            key = '{}-{}'.format(domain, slot)
            if key not in state_gathered:
                state_gathered[key] = ''
            if len(state_gathered[key]) > 0 and state_gathered[key][-1] == state:
                continue
            state_gathered[key] += state + ' ';

        for dontcare in dontcare_out:
            s, d = dontcare
            domain = domain_dicrionary.index2word[d]
            slot = slot_dictionary.index2word[s]
            domain = domain.replace("_DOMAIN", "")
            slot = slot.replace("_SLOT", "")
            key = '{}-{}'.format(domain, slot)
            state_gathered[key] = 'dontcare'

        predicted_state = []
        for k,v in state_gathered.items():
            predicted_state.append('{}-{}'.format(k, v.strip()))

        label_state = []
        for s in belief_state:
            v = s.split('-')[-1]
            if v != 'none':
                label_state.append(s)

        if dialogue_id not in curr_prediction:
            curr_prediction[dialogue_id] = {}
        if turn_id not in curr_prediction[dialogue_id]:
            curr_prediction[dialogue_id][turn_id] = {}

        item = {}
        item['turn_belief'] = sorted(label_state)
        item['predicted_belief'] = predicted_state
        curr_prediction[dialogue_id][turn_id] = item
    return curr_prediction
