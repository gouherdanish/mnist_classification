import torch
import torchvision 

from prep.base_prepare import DataPreparation
from data.incremental_data import IncrementalDataset
from custom.transforms import InvertIntensity

class IncrementalTestDataPreparation(DataPreparation):
    def __init__(self,test_image_path) -> None:
        self.test_image_path = test_image_path
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            InvertIntensity(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ])

    def _load_data(self,test_image_path):
        return IncrementalDataset(
            image_path=test_image_path,
            transform=self.transform
        )

    def prepare(self,**kwargs):
        test_ds = self._load_data(test_image_path=kwargs['test_image_path'])
        test_loader = torch.utils.data.DataLoader(
            dataset=test_ds, 
            shuffle=False, 
            batch_size=1)
        return test_loader