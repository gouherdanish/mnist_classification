from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision 

from constants import Constants, HyperParams

class DataPreparation(ABC):
    def __init__(self) -> None:
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5),(0.5))
        ])

    @abstractmethod
    def prepare(self):
        pass