import torch
import torch.nn as nn
import torchvision 

from constants import Constants, HyperParams
from base_prepare import DataPreparation

class TrainingDataPreparation(DataPreparation):
    def _load_data(self):
        self.train_ds = torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=self.transform
        )
    
    def _split_data(self):
        validation_split_frac = Constants.VALIDATION_SPLIT_FRAC
        val_size = int(validation_split_frac * len(self.train_ds))
        train_size = len(self.train_ds) - val_size
        self.train_ds, self.valid_ds = torch.utils.data.random_split(self.train_ds,(train_size,val_size))

    def prepare(self):
        self._load_data()
        self._split_data()
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_ds, 
            shuffle=True, 
            batch_size=HyperParams.BATCH_SIZE)
        val_loader = torch.utils.data.DataLoader(
            dataset=self.valid_ds, 
            shuffle=False, 
            batch_size=HyperParams.BATCH_SIZE)
        return train_loader, val_loader