import torch
import torch.nn as nn
import torchvision 

from constants import Constants, HyperParams
from base_prepare import DataPreparation

class TestDataPreparation(DataPreparation):
    def _load_data(self):
        self.test_ds = torchvision.datasets.MNIST(
            root="data",
            train=False,
            download=False,
            transform=self.transform
        )

    def prepare(self):
        self._load_data()
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_ds, 
            shuffle=False, 
            batch_size=HyperParams.BATCH_SIZE)
        return test_loader