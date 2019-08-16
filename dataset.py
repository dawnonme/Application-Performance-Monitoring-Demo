import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from mock_data import OUTPUT_PATH_TRAIN, OUTPUT_PATH_TEST, KPIS


class AutoXDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.kpi_df = pd.read_csv(csv_file)
        self.transform = transform
        self.vals = list(self.kpi_df.values)
        self.num_attr = len(KPIS)

    def __len__(self):
        return len(self.kpi_df)

    def __getitem__(self, idx):
        nd_array = self.vals[idx]
        inputs = torch.tensor(nd_array[1:self.num_attr + 1])
        target = torch.tensor(nd_array[self.num_attr + 2:]).squeeze()
        return inputs, target


def get_dataloader(data_type, batch_size, transform=None):
    if data_type == 'train':
        path = OUTPUT_PATH_TRAIN
    elif data_type == 'test':
        path = OUTPUT_PATH_TEST
    else:
        print('Invalid data type!')
        return None

    dataset = AutoXDataset(path, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)
    return dataloader
