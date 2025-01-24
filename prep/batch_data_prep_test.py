import torch
import torchvision 

from constants import HyperParams
from prep.base_prepare import DataPreparation
from data.batch_data import BatchData

class BatchTestDataPreparation(DataPreparation):
    def __init__(self) -> None:
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5),(0.5)),
            torchvision.transforms.CenterCrop(20)
        ])
        
    def _load_data(self):
        return BatchData(
            train=False,
            transform=self.transform
        ).load()

    def prepare(self):
        test_ds = self._load_data()
        test_loader = torch.utils.data.DataLoader(
            dataset=test_ds, 
            shuffle=False, 
            batch_size=HyperParams.BATCH_SIZE)
        return test_loader