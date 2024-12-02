import torch
import torchvision 

from prep.base_prepare import DataPreparation
from data.incremental_data import IncrementalDataset
from custom.transforms import InvertIntensity

class IncrementalDataPreparation(DataPreparation):
    def __init__(self,pil_img=None,image_path=None) -> None:
        self.pil_img = pil_img
        self.image_path = image_path
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            InvertIntensity(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ])

    def _load_data(self):
        return IncrementalDataset(
            pil_img=self.pil_img,
            image_path=self.image_path,
            transform=self.transform
        )

    def prepare(self):
        dataset = self._load_data()
        loader = torch.utils.data.DataLoader(
            dataset=dataset, 
            shuffle=False, 
            batch_size=1)
        return loader