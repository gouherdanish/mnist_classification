import torch
import torchvision 

from constants import HyperParams
from prep.base_prepare import DataPreparation
from data.batch_data import BatchData

class BatchTestDataPreparation(DataPreparation):
    def __init__(self) -> None:
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5),(0.5))
        ])
        
    def _load_data(self):
        self.test_ds = BatchData(
            train=False,
            transform=self.transform
        ).load()

    def prepare(self):
        self._load_data()
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_ds, 
            shuffle=False, 
            batch_size=HyperParams.BATCH_SIZE)
        return test_loader