import torch
import torch.nn as nn
import torchvision 

from constants import Constants, HyperParams
from data.base_prepare import DataPreparation

class BatchTestDataPreparation(DataPreparation):
    def __init__(self) -> None:
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5),(0.5))
        ])
        
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