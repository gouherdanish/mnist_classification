import torch
import torchvision 

from prep.base_prepare import DataPreparation
from data.incremental_data import IncrementalDataset

class InvertIntensity:
    def __call__(self, image):
        # Assuming the image is a PyTorch tensor with values in the range [0, 1]
        return 1 - image

class IncrementalTestDataPreparation(DataPreparation):
    def __init__(self,test_image_path) -> None:
        self.test_image_path = test_image_path
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            InvertIntensity(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ])

    def _load_data(self):
        self.test_ds = IncrementalDataset(
            image_path=self.test_image_path,
            transform=self.transform
        )

    def prepare(self):
        self._load_data()
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_ds, 
            shuffle=False, 
            batch_size=1)
        return test_loader