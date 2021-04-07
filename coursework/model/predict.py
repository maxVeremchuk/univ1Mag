import numpy as np
from config import *


def predict(data, model, dictionary, domain_dicrionary, slot_dictionary, curr_prediction):
    sorted_index = np.argsort(data['turn_id'])
    print(data)
    for i in sorted_index:
        curr_item_info = {}
        for k, v in data.items():
            curr_item_info[k] = v[i]  # i - batch position in value list

        # out, generated_fetrility, generated_gates = model.fertility_decoder(
        #     curr_item_info)  # first encoder decoder

        # slots_out, domains_out, dontcare_out = get_state_encoder_inport_from_generated_fertility(
        #     generated_fetrility, generated_gates, curr_item_info["slots"], curr_item_info["domains"])

        # if len(slots_out):
        #     print("len slots out is 0---------------")
        #     print(slots_out)
        #     print(domains_out)
        #     print(dontcare_out)
        #     print("-"*20)

        # out = model.state_decoder(out)  # second decoder

        # curr_prediction = generate_prediction(curr_item_info, slots_out, domains_out,
        #                                  dontcare_out, dictionary, domain_dicrionary, slot_dictionary, curr_prediction)
        curr_prediction = generate_prediction(curr_item_info, None, None,
                                              None, None, None, None, None)

    return curr_prediction


def get_state_encoder_inport_from_generated_fertility(fertilities, gates, slots, domains):
    dict_value_out = {}
    dontcare_out = []
    for i in range(fertilities.shape[0]):
        fertility = fertilities[i]
        gate = gates[i]
        slot = slots[i]
        domain = domains[i]
        if gate == GATES['none']:
            continue
        elif gate == GATES['dontcare']:
            dontcare_out.append((slot, domain))
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


def generate_prediction(curr_item_info, slots_out, domains_out,
                        dontcare_out, dictionary, domain_dicrionary, slot_dictionary, curr_prediction):
    dialogue_id = curr_item_info['dialogue_id']
    turn_id = curr_item_info['turn_id']
    print(curr_item_info)
    return []
