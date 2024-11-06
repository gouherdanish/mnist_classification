from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

from constants import Constants, HyperParams

class DataPrepation:
    def __init__(self) -> None:
        self.transform = transforms.Compose(
            transforms.ToTensor(),
            transforms.Normalize((0.5),(0.5))
        )
        self._load_data()
        self._split_data()

    def _load_data(self):
        self.train_ds = datasets.MNIST(
            root="data",
            train=True,
            download=False,
            transform=self.transform
        )
        self.test_ds = datasets.MNIST(
            root="data",
            train=False,
            download=False,
            transform=self.transform
        )
    
    def _split_data(self):
        validation_split_frac = Constants.VALIDATION_SPLIT_FRAC
        val_size = int(validation_split_frac * len(self.train_ds))
        train_size = len(self.train_ds) - val_size
        self.train_ds, self.valid_ds = random_split(self.train_ds,(train_size,val_size))

    def prepare(self):
        train_loader = DataLoader(
            dataset=self.train_ds, 
            shuffle=True, 
            batch_size=HyperParams.BATCH_SIZE)
        val_loader = DataLoader(
            dataset=self.valid_ds, 
            shuffle=False, 
            batch_size=HyperParams.BATCH_SIZE)
        test_loader = DataLoader(
            dataset=self.test_ds, 
            shuffle=False, 
            batch_size=HyperParams.BATCH_SIZE)
        return train_loader, val_loader, test_loader