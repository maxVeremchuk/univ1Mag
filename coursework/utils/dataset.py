import torch
import torch.utils.data as data

from config import *

class Dataset(data.Dataset):
    def __init__(self, data_info, global_dictionary, domain_dictionary, slot_dictionay):
        self.data_info_list = data_info
        self.global_dictionary = global_dictionary
        self.domain_dictionary = domain_dictionary
        self.slot_dictionay = slot_dictionay

    def __getitem__(self, index):
        data_info = self.data_info_list[index]
        id = data_info['id']
        turn_id = data_info['turn_id']
        dialogue_history = self.convert_to_index(data_info['dialogue_history'], self.global_dictionary.word2index)
        delex_dialogue_history = self.convert_to_index(data_info['delex_dialogue_history'], self.global_dictionary.word2index)
        all_domains = self.convert_to_index(data_info['domains'], self.domain_dictionary.word2index)
        all_slots = self.convert_to_index(data_info['slots'], self.slot_dictionay.word2index)
        domain_fertility = self.convert_to_index(data_info['domain_fertility'], self.domain_dictionary.word2index)
        slots_fertility = self.convert_to_index(data_info['slots_fertility'], self.slot_dictionay.word2index)
        slot_values = self.convert_to_index(data_info['slot_values'], self.global_dictionary.word2index)
        turn_belief = data_info['turn_belief']
        slots_domains = data_info['slots_domains']
        turn_belief_dict = data_info['turn_belief_dict']
        turn_utterance = self.convert_to_index(data_info['turn_utterance'], self.global_dictionary.word2index)
        fertility = data_info['fertility']
        gates = data_info['gates']

        item_info = {
            "dialogue_id": id,
            "turn_id": turn_id,
            "dialogue_history": dialogue_history,
            "delex_dialogue_history": delex_dialogue_history,
            "domains": all_domains,
            "slots": all_slots,
            "domain_fertility": domain_fertility,
            "slots_fertility": slots_fertility,
            "slot_values": slot_values,
            "turn_belief": turn_belief,
            "slots_domains": slots_domains,
            "turn_belief_dict": turn_belief_dict,
            "turn_utterance": turn_utterance,
            "fertility": fertility,
            "gates": gates
        }
        return item_info


    def convert_to_index(self, seq, dictionary_w2i):
        indexed_seq = []
        for word in seq:
            indexed_seq.append(dictionary_w2i[word] if word in dictionary_w2i else UNK_token_id)
        indexed_seq = torch.Tensor(story)
        return indexed_seq

def collate_fn(data):
    def merge(sequences, pad_token, max_len=-1):
        lengths = [len(seq) for seq in sequences]
        if max_len < 0:
            max_len = 1 if max(lengths)==0 else max(lengths)

        padded_seqs = torch.ones(len(sequences), max_len).long() * pad_token
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if type(seq) == list:
                padded_seqs[i, :end] = torch.Tensor(seq[:end])
            else:
                padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach()
        return padded_seqs

    data.sort(key=lambda x: len(x['dialogue_history']), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]


    dialogue_history = merge(item_info['dialogue_history'], PAD_token_id)
    delex_dialogue_history = merge(item_info['delex_dialogue_history'], PAD_token_id)
    domain_fertility = merge(item_info['domain_fertility'], PAD_token_id)
    slots_fertility = merge(item_info['slots_fertility'], PAD_token_id)
    slot_values = merge(item_info['slot_values'], PAD_token_id)
    fertility = merge(item_info['fertility'], PAD_token_id)
    gates = merge(item_info['gates'], GATES['none'])

    item_info['dialogue_history'] = dialogue_history
    item_info['delex_dialogue_history'] = delex_dialogue_history
    item_info['domain_fertility'] = domain_fertility
    item_info['slots_fertility'] = slots_fertility
    item_info['slot_values'] = slot_values
    item_info['fertility'] = fertility
    item_info['gates'] = gates

    return item_info

