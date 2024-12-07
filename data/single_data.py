import torch
import numpy as np
from torch.utils.data import Dataset

class SingleDataset(Dataset):
    def __init__(
            self,
            img:np.ndarray,
            label:int,
            transform=None) -> None:
        self.img = img
        self.label = label
        self.transform = transform

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        if self.transform: 
            img = self.transform(self.img)
        return img, torch.tensor(self.label)