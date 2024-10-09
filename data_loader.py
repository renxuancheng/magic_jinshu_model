from typing import Iterator
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import struct
import yaml
from yaml.loader import SafeLoader

DATA_HEADER_SIZE = 8


class JinshuDataset(Dataset):
    def __init__(self, config, data_path, block_size):
        self.config = config
        self.data_path = data_path
        with open(self.data_path, "rb") as f_read:
            self.arr_len = struct.unpack("<Q", f_read.read(DATA_HEADER_SIZE))
            print(self.arr_len)
        self.block_size = block_size + 1
        self.data_array = np.memmap(
            self.data_path, dtype=np.uint32, offset=DATA_HEADER_SIZE, shape=(self.arr_len[0], self.block_size)
        )

    def __len__(self):
        return self.arr_len[0]

    def __getitem__(self, index):
        return self.data_array[index,]


if __name__ == "__main__":
    print(0)
    with open("./config/magic_jinshu_pt.yaml", "r", encoding="utf-8") as f_read:
        config = yaml.load(f_read, SafeLoader)
        jin_shu_dataset = JinshuDataset(
            config, data_path=config["data"]["train_path"], block_size=config["model"]["block_size"] + 1
        )
        jin_shu_dataloader = DataLoader(jin_shu_dataset, batch_size=2)
