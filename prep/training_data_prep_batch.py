import torch
import torchvision 

from constants import DataConstants, HyperParams
from prep.base_prepare import DataPreparation
from data.batch_data import BatchData

class BatchTrainingDataPreparation(DataPreparation):
    def __init__(self) -> None:
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5),(0.5))
        ])

    def _load_data(self):
        return BatchData(
            train=True,
            transform=self.transform
        ).load()
    
    def _split_data(self,train_ds):
        validation_split_frac = DataConstants.VALIDATION_SPLIT_FRAC
        val_size = int(validation_split_frac * len(train_ds))
        train_size = len(train_ds) - val_size
        train_ds, valid_ds = torch.utils.data.random_split(train_ds,(train_size,val_size))
        return train_ds, valid_ds

    def prepare(self):
        train_ds = self._load_data()
        train_ds, valid_ds = self._split_data(train_ds)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds, 
            shuffle=True, 
            batch_size=HyperParams.BATCH_SIZE)
        val_loader = torch.utils.data.DataLoader(
            dataset=valid_ds, 
            shuffle=False, 
            batch_size=HyperParams.BATCH_SIZE)
        return train_loader, val_loader