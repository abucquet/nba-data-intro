"""
This file contains custom pytorch utility classes:
		- a custom Dataset class to make a pytorch Dataset from numpy arrays
"""
from torch.utils.data import Dataset
from torch import tensor, float32
#import torch.float as torch_float


class DatasetFromNumpy(Dataset):
    def __init__(self, x, y):
        super(DatasetFromNumpy, self).__init__()
        assert x.shape[0] == y.shape[0], (x.shape, y.shape) # assuming shape[0] = dataset size
        self.x = tensor(x, dtype=float32)
        self.y = tensor(y, dtype=float32)
        self.size = y.shape[0]


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.x[index], self.y[index]