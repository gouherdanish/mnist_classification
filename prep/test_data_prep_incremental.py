import torch
import torch.nn as nn
import torchvision 
from torch.utils.data import Dataset
from PIL import Image

from constants import Constants, HyperParams
from prep.base_prepare import DataPreparation

class SingleImageDataset(Dataset):
    def __init__(self,image_path) -> None:
        self.image_path = image_path
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ])

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        img = Image.open(self.image_path).convert("L")
        if self.transform: 
            img = self.transform(img)
        return img

class IncrementalTestDataPreparation(DataPreparation):
    def __init__(self,test_image_path) -> None:
        super().__init__()
        self.test_image_path = test_image_path

    def _load_data(self):
        self.test_ds = SingleImageDataset(image_path=self.test_image_path,transform=self.transform)

    def prepare(self):
        self._load_data()
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_ds, 
            shuffle=False, 
            batch_size=1)
        return test_loader