import json

from dictionary import *
from config import *
from dataset import *


# def extract_slot_domain_info(ontology):
#     slots_domains = sorted([k.replace(" ", "").lower()
#                             for k, v in ontology.items()])
#     domains = [item.split("-")[0] for item in slots_domains]
#     slots = [item.split("-")[1] for item in slots_domains]
#     # domains = list(dict.fromkeys(domains))  # remove duplicates
#     # slots = list(dict.fromkeys(slots))  # remove duplicates
#     return slots_domains, slots, domains


def extract_fertility_and_gates(slots_dict, slots_domains):
    fertility = [0] * len(slots_domains)
    gates = [GATES["none"]] * len(slots_domains)
    for k, v in slots_dict.items():
        index = slots_domains.index(k)
        fertility[index] = len(v.split())  # number of words in slot-value
        if v not in ['dontcare', 'none']:
            gates[index] = GATES["gen"]
        else:
            gates[index] = GATES[v]
    return fertility, gates


def extract_curr_slot_domain_fertility(slots_domains, fertility, curr_turn_slots_dict):
    domains = []
    slots = []
    values = []
    for idx, words_num in enumerate(fertility):
        if words_num == 0:
            continue
        domain = slots_domains[idx].split("-")[0]
        slot = slots_domains[idx].split("-")[1]
        value = curr_turn_slots_dict[slots_domains[idx]].split()
        for i in words_num:
            domains.append(domain+"_DOMAIN")
            slots.append(slot+"_SLOT")
            values.append(value[i])
    return domains, slots, values


def generate_data_detail(file_name, slots_domains, global_dict, domain_dict, slot_dict):
    data = []
    with open(file_name) as f:
        dials = json.load(f)

        for dialogue_data in dials:
            dialogue_history = ""
            delex_dialogue_history = ""
            for dialogue_turn in dialogue_data["dialogue"]:

                all_domains = [item.split(
                    "-")[0]+"_DOMAIN" for item in slots_domains]
                all_slots = [item.split(
                    "-")[1]+"_SLOT" for item in slots_domains]

                domain_dict.index_words(all_domains)
                slot_dict.index_words(all_slots)

                global_dict.index_words(all_domains)
                global_dict.index_words(all_slots)

                global_dict.index_words(dialogue_turn["transcript"])
                global_dict.index_words(dialogue_turn["system_transcript"])
                global_dict.index_words(dialogue_turn["delex_transcript"])
                global_dict.index_words(
                    dialogue_turn["delex_system_transcript"])

                user_turn = SOS_token + dialogue_turn["transcript"] + EOS_token
                system_turn = SOS_token + \
                    dialogue_turn["system_transcript"] + EOS_token
                delex_user_turn = SOS_token + \
                    dialogue_turn["delex_transcript"] + EOS_token
                delex_system_turn = SOS_token + \
                    dialogue_turn["delex_system_transcript"] + EOS_token

                dialogue_history = system_turn + user_turn
                delex_dialogue_history = delex_system_turn + delex_user_turn

                curr_turn_slots_dict = dict([(l["slots"][0][0].replace(" ", ""), l["slots"][0][1])
                                             for l in dialogue_turn["belief_state"]])  # get curr slot-domain info

                fertility, gates = extract_fertility_and_gates(
                    curr_turn_slots_dict, slots_domains)

                domain_fertility, slots_fertility, slot_values = extract_curr_slot_domain_fertility(
                    slots_domains, fertility, curr_turn_slots_dict)

                curr_turn_slots_list = [str(k) + '-' + str(v)
                                        for k, v in curr_turn_slots_dict.items()]

                data_detail = {
                    "dialogue_id": dials["dialogue_idx"],
                    "turn_id": dialogue_turn["turn_idx"],
                    "dialogue_history": dialogue_history,
                    "delex_dialogue_history": delex_dialogue_history,
                    "domains": all_domains,
                    "slots": all_slots,
                    "domain_fertility": domain_fertility,
                    "slots_fertility": slots_fertility,
                    "slot_values": slot_values,
                    "turn_belief": curr_turn_slots_list,
                    "slots_domains": slots_domains,
                    "turn_belief_dict": curr_turn_slots_dict,
                    "turn_utterance": system_turn + user_turn,
                    "fertility": fertility,
                    "gates": gates
                }
                data.append(data_detail)
    return data


def create_data_loader(data_info, global_dict, domain_dict, slot_dict, batch_size, shuffle):
    dataset = Dataset(data_info, global_dict, domain_dict, slot_dict)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader


def prepare_data(args):
    ontology = json.load(open(onology_path, 'r'))
    # get all sorted slots and domains
    slots_domains = sorted([k.replace(" ", "").lower()
                            for k, v in ontology.items()])
    global_dict = Dictionary()
    domain_dict = Dictionary()
    slot_dict = Dictionary()

    # list of data details in each dialigue
    train_data = generate_data_detail(
        train_data_path, slots_domains, global_dict, domain_dict, slot_dict)
    dev_data = generate_data_detail(
        dev_data_path, slots_domains, global_dict, domain_dict, slot_dict)
    test_data = generate_data_detail(
        test_data_path, slots_domains, global_dict, domain_dict, slot_dict)

    train_data_loader = create_data_loader(
        train_data, global_dict, domain_dict, slot_dict, args["batch_size"], False)
    dev_data_loader = create_data_loader(
        dev_data, global_dict, domain_dict, slot_dict, args["batch_size"], False)
    test_data_loader = create_data_loader(
        test_data, global_dict, domain_dict, slot_dict, args["batch_size"], False)

    return train_data_loader, dev_data_loader, test_data_loader
