from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision 

from constants import Constants, HyperParams

class DataPreparation(ABC):
    @abstractmethod
    def prepare(self):
        pass