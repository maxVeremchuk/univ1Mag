import numpy as np
import torch
from config import *


def predict(data, model, dictionary, domain_dicrionary, slot_dictionary, curr_prediction):
    sorted_index = np.argsort(data['turn_id'])
    print(data)
    for i in sorted_index:
        curr_item_info = {}
        for k, v in data.items():
            curr_item_info[k] = v[i]  # i - batch position in value list
        #TODO add .unsqueeze(0)
        curr_item_info['domains'] = curr_item_info['domains'].unsqueeze(0)
        curr_item_info['slots'] = curr_item_info['slots'].unsqueeze(0)
        curr_item_info['context'] = curr_item_info['context'].unsqueeze(0)
        curr_item_info['delex_context'] = curr_item_info['delex_context'].unsqueeze(0)
        _, generated_fetrility, generated_gates = model.fertility_decoder.forward(
            curr_item_info)  # first encoder decoder

        generated_fetrility = generated_fetrility.squeeze(0)
        generated_gates = generated_gates.squeeze(0)

        slots_out, domains_out, dontcare_out = get_state_encoder_inport_from_generated_fertility(
            generated_fetrility, generated_gates, curr_item_info["slots"].squeeze(0), curr_item_info["domains"].squeeze(0))
        print("-"*10)
        print(curr_item_info["domains"])
        if len(slots_out) == 0:
            print("len slots out is 0---------------")
            print(slots_out)
            print(domains_out)
            print(dontcare_out)
            print("-"*20)

        slots_out = torch.stack(slots_out).long().unsqueeze(0)
        domains_out = torch.stack(domains_out).long().unsqueeze(0)

        curr_item_info["slots"] = slots_out
        curr_item_info["domains"] = domains_out
        out = model.state_decoder(curr_item_info)  # second decoder

        curr_prediction = generate_prediction(curr_item_info, generated_fetrility, generated_gates, slots_out, domains_out,
                                            dontcare_out, out['generated_y'], dictionary, domain_dicrionary, slot_dictionary, curr_prediction)

        # curr_prediction = generate_prediction(curr_item_info, None, None,
        #                                       None, None, None, None, None)

    return curr_prediction


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


def generate_prediction(curr_item_info, generated_fetrility, generated_gates, slots_out, domains_out,
                        dontcare_out, generated_states, dictionary, domain_dicrionary, slot_dictionary, curr_prediction):
    dialogue_id = curr_item_info['dialogue_id']
    turn_id = curr_item_info['turn_id']
    turn_belief_dict = curr_item_info['turn_belief_dict']

    print(generated_states)
    states = [dictionary.index2word[i] for i in generated_states.squeeze(0).tolist()]
    slots = [slot_dictionary.index2word[i] for i in slots_out.squeeze(0).tolist()]
    domains = [domain_dicrionary.index2word[i] for i in domains_out.squeeze(0).tolist()]

    state_gathered = {}
    for idx in range(len(domains)):
        state = states[idx]
        slot = slots[idx]
        domain = domains[idx]
        if 'PAD' in [domain, slot, state]: continue
        if 'EOS' in [domain, slot, state]: continue
        if 'SOS' in [domain, slot, state]: continue
        if 'UNK' in [domain, slot, state]: continue
        if 'dontcare' in [state]: continue
        if 'none' in [state]: continue

        slot = slot.replace("_SLOT", "")
        domain = domain.replace("_DOMAIN", "")
        key = '{}-{}'.format(domain,slot)
        if key not in state_gathered: state_gathered[key] = []
        if len(state_gathered[key])>0 and state_gathered[key][-1] == state: continue
        state_gathered[key].append(state)

    for dontcare in dontcare_out:
        s, d = dontcare
        domain = domain_dicrionary.index2word[d]
        slot = slot_dictionary.index2word[s]
        domain = domain.replace("_DOMAIN", "")
        slot = slot.replace("_SLOT", "")
        key = '{}-{}'.format(domain,slot)
        state_gathered[key] = 'dontcare'

    print(dialogue_id)
    if dialogue_id not in curr_prediction: curr_prediction[dialogue_id] = {}
    if turn_id not in curr_prediction[dialogue_id]: curr_prediction[dialogue_id][turn_id] = {}


    # item = {}
    # item['predicted_belief'] = state_gathered
    curr_prediction[dialogue_id][turn_id] = state_gathered
    return curr_prediction
