import torch
import torch.utils.data as data

from utils.config import *


class Dataset(data.Dataset):
    def __init__(self, data_info):
        self.data_info_list = data_info

    def __getitem__(self, index):
