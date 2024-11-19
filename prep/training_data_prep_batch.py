import torch
import torchvision 

from constants import Constants, HyperParams
from prep.base_prepare import DataPreparation
from data.batch_data import BatchData

class BatchTrainingDataPreparation(DataPreparation):
    def _load_data(self):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5),(0.5))
        ])
        self.train_ds = BatchData(
            train=True,
            transform=self.transform
        ).load()
    
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